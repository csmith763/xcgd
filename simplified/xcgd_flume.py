"""
Density-based Flume Topology Overview:
--------------------------------------

As a reference, the setup within Flume topology for performing a natural frequency analysis with density-based topology optimization can be broken down into classes.

1) TopoAnalysis (located at flume_topology/analyses/topo_analysis.py): this is a regular Python class, which serves as an interface between Python and C++. It wraps many functions within C++ that leverage Eigen, e.g. to assemble the stiffness matrix for linear quad/hexahedral elements. This is then provided as an argument during the construction of many other Flume topology objects so that those classes can access methods within the TopoAnalysis class, which encodes information about the mesh's node locations and element connectivity.

2) NodeFilter (located at flume_topology/analyses/node_filter.py): this is the source for the analysis procedures within Flume topology. It takes in the nodal design variables, x, and outputs the element-wise density field, rhoE (also computes the node-wise density field internally). Parameters (passed in as kwargs during construction) are used to control features like the filter type, characteristic length, and/or design variable mapping. The element-wise density field is then distributed everywhere else, such as for the frequency analysis, to compute the eigenvalues and eigenvectors. It is also used for many of the methods within the TopoAnalysis object to compute the finite element quantities of interest using the density field.

3) NaturalFrequencyAnalysis (located at flume_topology/analyses/frequency_analysis.py): this is the class that wraps eigd and is responsible for solving the eigenvalue problem. Ultimately, this class returns the eigenvalues and eigenvectors for use within the topology optimization problem.

4) Optional class for XCGD wrapper (e.g. stores the background mesh, cut_mesh, stiffness/mass assemblers, physics classes; basically just a dataclass storing other objects of interest)

XCGD-Flume Topology Proposed Structure
--------------------------------------

ParOpt/MMA owns the design variables x and the optimization loop still (other level-set approaches may use a velocity field, which is not a consistent approach with Flume topology).

Forward map
    x  ->  psi (filtered level-set function)  ->  cut mesh  ->  (K, M)  ->  lambda, Q  ->  objective/constraints

1) LevelSetFilter: applies the Helmholtz operator (assembled by HelmholtzAssembler on the background mesh) to map the nodal design variables x to a regularized nodal level-set field psi. This also allows for imposing the length scale during the topology optimization problem.

2) XCGDFrequencyAnalysis: builds the stiffness and mass assemblers on the interior mesh (create_interior_mesh), assembles K and M, solves the generalized eigenproblem with eigd, and forms the objective/constraints (e.g. a KS-aggregate of the lowest eigenvalues). Eigenvectors are mapped from the interior (renumbered) node set back to background-node indexing.

Adjoint / reverse map:

2') eigenvalue sensitivity: from d(obj)/d(lambda, Q), form the shape derivative of the objective with respect to psi using the eigenproblem sensitivity

1') LevelSetFilter (transpose): Reuse the same Helmholtz factorization from the forward solve, and propagate the derivatives back from the cut-cell mesh to the entire field of design variables. This consistent gradient is returned to ParOpt/MMA.

"""

from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
import xcgd as xd
from scipy.sparse import csr_matrix
from eigd import IRAM, make_operator


class LevelSetFilter(Analysis):

    def __init__(
        self, mesh: xd.CartesianMesh, obj_name: str, sub_analyses=[], **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {
            "r0": 1.0,  # characteristic length
            # projection parameters
            # any other parameters for mesh
        }

        # Store the mesh object
        self.mesh = mesh

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default variable values
        x_var = State(
            value=np.ones(mesh.nnodes),
            desc="Design variables for the structure",
            source=self,
        )

        self.variables = {"x": x_var}

        # Initialize the Helmholtz filter
        self._initialize_helmholtz()

        return

    def _initialize_helmholtz(self):
        """
        Construct the Helmholtz physics and assembler
        """

        return

    def _analyze(self):
        """
        Responsible for mapping the design variables to the level-set field (and applying the filter in the process).
        """

        # Store the outputs
        self.outputs = {}

        # self.outputs["psi"] = State(...)

        return

    def _analyze_adjoint(self):
        """
        Propagates derivatives backwards from the level-set function to the design variables for the problem.
        """

        return


class XCGDFrequencyAnalysis(Analysis):

    def __init__(
        self,
        mesh: xd.CartesianMesh,
        stiffness_assembler: xd.Assembler,
        mass_assembler: xd.Assembler,
        obj_name: str,
        sub_analyses=list[LevelSetFilter],
        **kwargs
    ):

        # Set the default parameters
        self.default_parameters = {
            # parameters for eigd and frequency analysis
        }

        # Store the mesh object
        self.mesh = mesh

        # Store the assembler objects
        self.K_assembler = stiffness_assembler
        self.M_assembler = mass_assembler

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default variable values
        lsf_var = State(
            value=np.ones(self.cut_mesh.nnodes),
            desc="Filtered level-set field",
            source=self,
        )

        self.variables = {"psi": lsf_var}

        # Setup the CartesianCutMesh at the first iteration
        self._setup_cut_mesh()

        return

    def _setup_cut_mesh(self):
        """
        Set up the Cartesian cut mesh object using the existing background mesh
        """

        self.cut_mesh = xd.CartesianCutMesh(self.mesh)

        return

    def _analyze(self):
        """
        Use the level-set field through the cut-cell class to update the stiffness/mass matrices, and then use eigd to solve the eigenvalue problem
        """

        # Extract the level-set function field
        psi = self.variables["psi"].value

        # Update the LSF for the cut cell object
        self.cut_mesh.get_lsf()[:] = psi.flatten()
        self.ut_mesh.update()

        # Update the sparsity patterns for the stiffness and mass matrices
        self.K_assembler.update()
        self.M_assembler.update()

        # # Evaluate the residual and the Jacobian
        self.K.eval_jacobian()
        self.M.eval_jacobian()

        # Retrieve the Jacobian we just computed
        kcsr = self.K_assembler.get_jacobian()
        mcsr = self.M_assembler.get_jacobian()

        # Construct SciPy CSR matrices
        K_sp = csr_matrix(
            (kcsr.data, kcsr.cols, kcsr.rowp), shape=(kcsr.nrows, kcsr.nrows)
        )
        M_sp = csr_matrix(
            (mcsr.data, mcsr.cols, mcsr.rowp), shape=(mcsr.nrows, mcsr.nrows)
        )

        # Compute the shifted operator
        sigma = self.parameters["sigma"]
        mat = K_sp - sigma * M_sp
        mat = (mat + mat.T) * 0.5

        # Construct the operator
        factor = make_operator(mat)

        # Construct the eigensolver
        N = self.parameters["N"]
        eig_atol = self.parameters["eig_atol"]
        tol = self.parameters["tol"]

        m = max(2 * N + 1, 60)
        eig_solver = IRAM(N=N, m=m, eig_atol=eig_atol, tol=tol)

        # Solve the eigenvalue problem
        lam, Q = eig_solver.solve(A=K_sp, B=M_sp, factor=factor, sigma=sigma)

        # Store the outputs in the dictionary
        self.outputs = {}

        self.outputs["lam"] = State(
            value=lam,
            desc="Array of the eigenvalues for the topology object",
            source=self,
        )

        self.outputs["Q"] = State(
            value=Q,
            desc="2D array that contains the eigenvectors for the topology object",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Responsible for propagating the derivatves from the eigenvalues/eigenvectors back through to the level-set function field.
        """

        return
