import xcgd
import numpy as np
from scipy.sparse import csr_matrix
from eigd import IRAM, make_operator
from icecream import ic
from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import matplotlib.pyplot as plt


class XCGDAnalysis:

    def __init__(self, lsf_tree, mesh_length, E=1.0, nu=0.3, rho=1.0):

        # Construct the quadtree mesh that defines the LSF
        self.lsf_tree = lsf_tree
        self.mesh_length = mesh_length
        self.lsf_mesh = xcgd.QuadtreeMesh(self.lsf_tree, mesh_length)

        # Create a quadtree for the mesh
        self.tree = self.lsf_tree.duplicate()
        self.tree.refine()
        self.tree.balance()
        self.mesh = xcgd.QuadtreeMesh(self.tree, mesh_length)

        # Set the cut mesh
        self.cut_mesh = xcgd.QuadtreeCutMesh(self.mesh, self.lsf_mesh)

        # Record the lsf coordinates
        X = np.array(self.lsf_mesh.get_node_locations())
        self.lsf_x = X[0::2]
        self.lsf_y = X[1::2]

        self.lsf_size = self.lsf_x.size

        # Allocate the problem physics
        self.E = E
        self.nu = nu
        self.rho = rho

        self.elas = xcgd.LinearElasticity2D(self.E, self.nu)
        self.mass = xcgd.ElasticityMass2D(self.rho)
        self.area = xcgd.Area2D()

        self.stiffness_assembler = None
        self.mass_assembler = None
        self.area_assembler = None
        self.perimeter_assembler = None

        return

    def _update_after_design_change(self):
        # Update the underlying quadtree mesh to reflect the balance changes
        self.mesh.update()

        # Update the cut mesh and it's derivatives
        self.cut_mesh.update()
        self.cut_mesh.update_derivatives()

        # Get the interior mesh
        interior_mesh = self.cut_mesh.create_interior_mesh()

        # Re-allocate the stiffness and mass assemblers
        self.stiffness_assembler = xcgd.Assembler(
            [xcgd.LinearElasticity2DAssembler(interior_mesh, self.elas)]
        )
        self.mass_assembler = xcgd.Assembler(
            [xcgd.ElasticityMass2DAssembler(interior_mesh, self.mass)]
        )

        # Re-allocate the area and perimeter assemblers using the interface mesh
        interface_mesh = self.cut_mesh.create_interface_mesh()
        self.area_assembler = xcgd.Assembler(
            [xcgd.Area2DAssembler(interior_mesh, self.area)]
        )
        self.perimeter_assembler = xcgd.Assembler(
            [xcgd.Area2DAssembler(interface_mesh, self.area)]
        )

        return

    def set_lsf(self, lsf):
        self.cut_mesh.get_lsf()[:] = lsf

        self._update_after_design_change()
        return

    def get_lsf_coordinates(self):
        return self.lsf_x, self.lsf_y

    def get_lsf_size(self):
        return self.lsf_size


class XCGDMinFrequency(Analysis):

    def __init__(
        self, xcgd_analysis: XCGDAnalysis, obj_name: str, sub_analyses=[], **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {
            "N": 13,
            "sigma": -0.1,
            "solver_type": "IRAM",
            "eig_atol": 1e-10,
            "tol": 1e-14,
            # "E": 1.0,
            # "nu": 0.3,
            # "rho": 1.0,
            "ks_param": 20.0,
            # "mesh_length": 1.0,
        }

        # Store the XCGDAnalysis object
        self.xcgd = xcgd_analysis

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        self.eig_solver = None

        # Get the size for the LSF mesh field
        lsf_size = self.xcgd.get_lsf_size()

        # Set the default variable values
        lsf_var = State(
            value=np.ones(lsf_size),
            desc="Filtered level-set field",
            source=self,
        )

        self.variables = {"lsf": lsf_var}

        return

    def solve_frequency_problem(
        self,
        stiffness_assembler,
        mass_assembler,
        sigma,
        N,
        tol,
        eig_atol,
    ):
        """
        Solves the natural frequency problem using eigd with Galerkin-difference for the numerical solution.
        """

        # Update the sparsity patterns for the stiffness and mass matrices
        stiffness_assembler.update()
        mass_assembler.update()

        # # Evaluate the residual and the Jacobian
        stiffness_assembler.eval_jacobian()
        mass_assembler.eval_jacobian()

        # Retrieve the Jacobian we just computed
        kcsr = stiffness_assembler.get_jacobian()
        mcsr = mass_assembler.get_jacobian()

        # Construct SciPy CSR matrices
        K_sp = csr_matrix(
            (kcsr.data, kcsr.cols, kcsr.rowp), shape=(kcsr.nrows, kcsr.nrows)
        )
        M_sp = csr_matrix(
            (mcsr.data, mcsr.cols, mcsr.rowp), shape=(mcsr.nrows, mcsr.nrows)
        )

        # Compute the shifted operator
        mat = K_sp - sigma * M_sp
        mat = (mat + mat.T) * 0.5

        # Construct the operator
        factor = make_operator(mat)

        # Construct the eigensolver
        m = max(2 * N + 1, 60)
        if self.eig_solver is None:
            self.eig_solver = IRAM(N=N, m=m, eig_atol=eig_atol, tol=tol)

        # Solve the eigenvalue problem
        lam, Q = self.eig_solver.solve(A=K_sp, B=M_sp, factor=factor, sigma=sigma)

        return lam, Q

    def get_lsf_coordiantes(self):

        # Record the lsf coordinates
        X = np.array(self.lsf_mesh.get_node_locations())
        self.lsf_x = X[0::2]
        self.lsf_y = X[1::2]

        return self.lsf_x, self.lsf_y

    def _analyze(self):

        # Extract the LSF function variables
        lsf = self.variables["lsf"].value

        # Set the LSF function values for the gut mesh object
        self.xcgd.set_lsf(lsf=lsf)  # FIXME: move this into the DV analysis class

        # Extract the various parameters
        sigma = self.parameters["sigma"]
        N = self.parameters["N"]
        tol = self.parameters["tol"]
        eig_atol = self.parameters["eig_atol"]

        K_assembler = self.xcgd.stiffness_assembler
        M_assembler = self.xcgd.mass_assembler

        # Solve the frequency problem with the updated values of the LSF
        lam, self.phi = self.solve_frequency_problem(
            stiffness_assembler=K_assembler,
            mass_assembler=M_assembler,
            sigma=sigma,
            N=N,
            tol=tol,
            eig_atol=eig_atol,
        )

        # Remove the rigid-body modes and store the remaining frequency
        self.frequency = np.sqrt(lam[3:])

        # Compute the KS min frequency
        min_freq = self.frequency[0]
        rho = self.parameters["ks_param"]
        ks = min_freq - np.log(np.sum(np.exp(-rho * (self.frequency - min_freq)))) / rho

        # Store the outputs
        self.outputs = {}

        self.outputs["omega_ks_min"] = State(
            value=ks,
            desc="Minimum natural frequency computed using the KS function from the XCGD frequency analysis",
            source=self,
        )

        return

    def _analyze_adjoint(self):

        # Extract the existing derivatives
        omega_ks_minb = self.outputs["omega_ks_min"].deriv
        lsfb = self.variables["lsf"].deriv

        # Compute the weighting for each eigenvalue derivative
        min_freq = self.frequency[0]
        rho = self.parameters["ks_param"]
        eta = np.exp(-rho * (self.frequency - min_freq))
        sum = np.sum(eta)
        eta = eta / sum

        # Compute the derivative of the objective wrt the eigenvalues
        dfdlam = omega_ks_minb * 0.5 * eta / self.frequency

        # Derivative wrt the lsf values
        dfdx = np.zeros(self.xcgd.get_lsf_size())

        stiffness_assembler = self.xcgd.stiffness_assembler
        mass_assembler = self.xcgd.mass_assembler

        for i, freq in enumerate(self.frequency):
            stiffness_assembler.get_dof()[:] = self.phi[:, i + 3]
            stiffness_assembler.zero_derivative()
            stiffness_assembler.add_functional_derivative()
            dfdx += dfdlam[i] * stiffness_assembler.get_dfdx()

            mass_assembler.get_dof()[:] = self.phi[:, i + 3]
            mass_assembler.zero_derivative()
            mass_assembler.add_functional_derivative()
            dfdx -= freq**2 * dfdlam[i] * mass_assembler.get_dfdx()

        dfdx *= 2.0

        # Update the derivative of the function wrt the LSF values
        lsfb += dfdx
        self.variables["lsf"].set_deriv_value(lsfb)

        return


class XCGDArea(Analysis):

    def __init__(
        self, xcgd_analysis: XCGDAnalysis, obj_name: str, sub_analyses=[], **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {}

        # Store the XCGDAnalysis object
        self.xcgd = xcgd_analysis

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Get the size for the LSF mesh field
        lsf_size = self.xcgd.get_lsf_size()

        # Set the default variable values
        lsf_var = State(
            value=np.ones(lsf_size),
            desc="Filtered level-set field",
            source=self,
        )

        self.variables = {"lsf": lsf_var}

        return

    def _analyze(self):

        # Extract the LSF function variables
        lsf = self.variables["lsf"].value

        # Set the LSF function values for the gut mesh object
        self.xcgd.set_lsf(lsf=lsf)  # FIXME: move this into the DV analysis class

        # Evaluate the area of the interior portion of the cut mesh
        area = self.xcgd.area_assembler.eval_functional()

        # Store the outputs
        self.outputs = {}

        self.outputs["area"] = State(
            value=area,
            desc="Area of the interior mesh defined by the LSF function",
            source=self,
        )

        return

    def _analyze_adjoint(self):

        # Extract the existing derivatives
        areab = self.outputs["area"].deriv
        lsfb = self.variables["lsf"].deriv

        # Evaluate the area derivative
        self.xcgd.area_assembler.zero_derivative()
        self.xcgd.area_assembler.add_functional_derivative()

        lsfb += areab * self.xcgd.area_assembler.get_dfdx()

        # Update the derivative value
        self.variables["lsf"].set_deriv_value(deriv_val=lsfb)

        return


class XCGDPerimeter(Analysis):

    def __init__(
        self, xcgd_analysis: XCGDAnalysis, obj_name: str, sub_analyses=[], **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {}

        # Store the XCGDAnalysis object
        self.xcgd = xcgd_analysis

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Get the size for the LSF mesh field
        lsf_size = self.xcgd.get_lsf_size()

        # Set the default variable values
        lsf_var = State(
            value=np.ones(lsf_size),
            desc="Filtered level-set field",
            source=self,
        )

        self.variables = {"lsf": lsf_var}

        return

    def _analyze(self):

        # Extract the LSF function variables
        lsf = self.variables["lsf"].value

        # Set the LSF function values for the gut mesh object
        self.xcgd.set_lsf(lsf=lsf)  # FIXME: move this into the DV analysis class

        # Evaluate the area of the interior portion of the cut mesh
        perimeter = self.xcgd.perimeter_assembler.eval_functional()

        # Store the outputs
        self.outputs = {}

        self.outputs["perimeter"] = State(
            value=perimeter,
            desc="perimeter of the cut-mesh defined by the LSF function",
            source=self,
        )

        return

    def _analyze_adjoint(self):

        # Extract the existing derivatives
        pb = self.outputs["perimeter"].deriv
        lsfb = self.variables["lsf"].deriv

        # Evaluate the area derivative
        self.xcgd.perimeter_assembler.zero_derivative()
        self.xcgd.perimeter_assembler.add_functional_derivative()

        lsfb += pb * self.xcgd.perimeter_assembler.get_dfdx()

        # Update the derivative value
        self.variables["lsf"].set_deriv_value(deriv_val=lsfb)

        return


class Starfish(Analysis):

    def __init__(
        self, xcgd_analysis: XCGDAnalysis, obj_name: str, sub_analyses=[], **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {
            "ndvs": 3,
            "circle_radius": 1.0,
            "mesh_length": 3.0,
            "starting_harmonic": 3.0,
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Store the XCGDAnalysis object
        self.xcgd = xcgd_analysis

        # Get the coordinates for the background LSF mesh
        self.x, self.y = self.xcgd.get_lsf_coordinates()

        # Define the baseline, circular LSF
        length = self.parameters["mesh_length"]
        x0 = y0 = length / 2

        # Compute the baseline LSF values for a circle
        self.baseline_lsf = (
            (self.x - x0) ** 2
            + (self.y - y0) ** 2
            - self.parameters["circle_radius"] ** 2
        )

        # Define the angular positions
        self.theta = np.atan2(self.y - y0, self.x - x0)
        # ic(self.theta)

        # Store the XCGDAnalysis object
        self.xcgd = xcgd_analysis

        # Set the default variable values
        coeff_vals = np.random.uniform(low=-0.1, high=0.1, size=self.parameters["ndvs"])

        coeff_vars = State(
            value=coeff_vals,
            desc="Coefficient values that serve as the design variables for creating the 'starfish' LSF",
            source=self,
        )

        self.variables = {"coeffs": coeff_vars}

        return

    def _analyze(self):

        # Extract the values of the design variables
        coeffs = self.variables["coeffs"].value

        # Get the baseline LSF values
        lsf = self.baseline_lsf.copy()

        # Loop through the number of design variables and add the contributions to the LSF
        starting_harmonic = self.parameters["starting_harmonic"]

        for i, coeff in enumerate(coeffs):
            # Compute the current harmonic index
            harmonic = starting_harmonic + i

            # Compute the contribution to add to the LSF values
            update = coeff * np.cos(harmonic * self.theta)

            # Add the contribution to the LSF values
            lsf += update

        # Set the updated value of the LSF function in the XCGDAnalysis object
        self.xcgd.set_lsf(lsf)

        # Store the updated value of the LSF value
        self.outputs = {}

        self.outputs["lsf"] = State(
            value=lsf, desc="Level-set function values", source=self
        )

        return

    def _analyze_adjoint(self):

        # Extract the existing derivatives
        coeffsb = self.variables["coeffs"].deriv
        lsfb = self.outputs["lsf"].deriv

        # Loop through and update the contributions to the coefficient derivatives
        starting_harmonic = self.parameters["starting_harmonic"]
        for i, cb in enumerate(coeffsb):
            # Compute the current harmonic index
            harmonic = starting_harmonic + i

            # Compute the contribution to cb
            cb_update = np.dot(lsfb, np.cos(harmonic * self.theta))

            # Update the cb value
            cb += cb_update
            coeffsb[i] = cb

        # Update the derivative values
        self.variables["coeffs"].set_deriv_value(deriv_val=coeffsb)

        return

    def plot_lsf(self):
        # Extract the value of the LSF function
        lsf = self.outputs["lsf"].value

        # Create a figure
        fig, ax = plt.subplots(1, 1)

        # Plot the contour
        # X, Y = np.meshgrid(self.x, self.y)
        # plt.tricontour(self.x, self.y, lsf, levels=4)
        ax.tricontour(self.x, self.y, lsf, levels=[0])

        ax.axis("equal")
        # ax.axis("off")

        return fig


if __name__ == "__main__":

    # Construct the source tree for the LSF function
    tree = xcgd.Quadtree()

    # Uniformly refine the mesh to have 2**5 = 32 elements along each edge
    tree.refine([4])
    tree.balance()

    tree.to_vtk("flume_tree.vtk")

    # Construct the XCGD analysis object
    xcgd_analysis = XCGDAnalysis(lsf_tree=tree, mesh_length=3.0)

    # Construct the design variable object
    starfish = Starfish(
        xcgd_analysis=xcgd_analysis, obj_name="starfish", sub_analyses=[], ndvs=10
    )

    # starfish.set_var_values(variables={"coeffs": np.array([0.25, 0.5])})
    starfish.analyze()
    fig = starfish.plot_lsf()

    # Test the adjoint for the starfish object
    # starfish.declare_design_vars(variables=["coeffs"])
    # starfish.test_isolated_adjoint(method="fd")

    # plt.show()

    # Construct the XCGDFrequencyAnalysisObject
    xcgd_freq = XCGDMinFrequency(
        xcgd_analysis=xcgd_analysis,
        obj_name="XCGD_Frequency",
        sub_analyses=[starfish],
    )

    starfish.declare_design_vars(variables=["coeffs"])
    xcgd_freq.test_combined_adjoint(method="fd", print_res=True)

    # Construct the XCGDArea object
    area = XCGDArea(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Area", sub_analyses=[starfish]
    )

    area.test_combined_adjoint(method="fd", print_res=True)

    # Construct the XCGDPerimeter object
    perimeter = XCGDPerimeter(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Perimeter", sub_analyses=[starfish]
    )

    perimeter.test_combined_adjoint(method="fd", print_res=True)

    # Set baseline values for the LSF
    # x0 = 1.5
    # y0 = 1.5
    # r0 = 1.0

    # x, y = xcgd_analysis.get_lsf_coordinates()

    # lsf = (x - x0) ** 2 + (y - y0) ** 2 - r0**2

    # xcgd_analysis.set_lsf(lsf)

    # # Test the adjoint for the XCGDMinFrequency object
    # xcgd_freq.declare_design_vars(variables=["lsf"])
    # xcgd_freq.test_isolated_adjoint(
    #     print_res=True, method="fd", defined_vars={"lsf": lsf}
    # )

    # # Test the adjoint for the XCGDArea objerct
    # area.declare_design_vars(variables=["lsf"])
    # area.test_isolated_adjoint(print_res=True, method="fd", defined_vars={"lsf": lsf})

    # # Test the adjoint for the XCGDPerimeter objerct
    # perimeter.declare_design_vars(variables=["lsf"])
    # perimeter.test_isolated_adjoint(
    #     print_res=True, method="fd", defined_vars={"lsf": lsf}
    # )
