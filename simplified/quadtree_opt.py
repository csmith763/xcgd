import xcgd
import numpy as np
import amigo as am
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix
from eigd import IRAM, make_operator
from icecream import ic

# from flume_topology.analyses.topo_analysis import TopoAnalysis
# from flume_topology.analyses.frequency_analysis import NaturalFrequencyAnalysis
# from flume_topology.utils.mesh_utils import create_beam_domain
import niceplots


def solve_frequency_problem(
    stiffness_assembler,
    mass_assembler,
    sigma=0.0,
    N=10,
    tol=1e-14,
    eig_atol=1e-5,
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
    K_sp = csr_matrix((kcsr.data, kcsr.cols, kcsr.rowp), shape=(kcsr.nrows, kcsr.nrows))
    M_sp = csr_matrix((mcsr.data, mcsr.cols, mcsr.rowp), shape=(mcsr.nrows, mcsr.nrows))

    # Compute the shifted operator
    mat = K_sp - sigma * M_sp
    mat = (mat + mat.T) * 0.5

    # Construct the operator
    factor = make_operator(mat)

    # Construct the eigensolver
    m = max(2 * N + 1, 60)
    eig_solver = IRAM(N=N, m=m, eig_atol=eig_atol, tol=tol)

    # Solve the eigenvalue problem
    lam, Q = eig_solver.solve(A=K_sp, B=M_sp, factor=factor, sigma=sigma)

    return lam, Q


class QuadtreeOpt:
    def __init__(self, lsf_tree, length=1.0, ks_param=20.0):
        self.length = length
        self.ks_param = ks_param

        # Quadtree that defines the lsf
        self.lsf_tree = lsf_tree
        self.lsf_mesh = xcgd.QuadtreeMesh(self.lsf_tree, self.length)

        # Create a quadtree for the mesh
        self.tree = self.lsf_tree.duplicate()
        self.tree.refine()
        self.tree.balance()
        self.mesh = xcgd.QuadtreeMesh(self.tree, self.length)

        # Set the cut mesh
        self.cut_mesh = xcgd.QuadtreeCutMesh(self.mesh, self.lsf_mesh)

        # Record the lsf coordinates
        X = np.array(self.lsf_mesh.get_node_locations())
        self.lsf_x = X[0::2]
        self.lsf_y = X[1::2]

        # Allocate the problem physics
        E, nu, rho = 1.0, 0.3, 1.0
        self.elas = xcgd.LinearElasticity2D(E, nu)
        self.mass = xcgd.ElasticityMass2D(rho)
        self.area = xcgd.Area2D()

        self.stiffness_assembler = None
        self.mass_assembler = None

    def _apply_refinement(self):
        interface_elems = self.cut_mesh.get_interface_elements()
        interior_elems = self.cut_mesh.get_interior_elements()

        # Specify the refinement level for the tree
        refinement = np.zeros(self.tree.size(), dtype=np.int32)
        refinement[interface_elems] = 2
        refinement[interior_elems] = 1
        self.tree.refine(refinement)
        self.tree.balance()

        return

    def _update_after_design_change(self):
        # Update the underlying quadtree mesh to reflect the balance changes
        self.mesh.update()

        # Update the cut mesh and it's derivatives
        self.cut_mesh.update()
        self.cut_mesh.update_derivatives()

        # Get the interior mesh
        interior_mesh = self.cut_mesh.create_interior_mesh()
        interface_mesh = self.cut_mesh.create_interface_mesh()

        # Re-allocate the stiffness assembler
        self.stiffness_assembler = xcgd.Assembler(
            [xcgd.LinearElasticity2DAssembler(interior_mesh, self.elas)]
        )
        self.mass_assembler = xcgd.Assembler(
            [xcgd.ElasticityMass2DAssembler(interior_mesh, self.mass)]
        )
        self.area_assembler = xcgd.Assembler(
            [xcgd.Area2DAssembler(interior_mesh, self.area)]
        )
        self.perimeter_assembler = xcgd.Assembler(
            [xcgd.Area2DAssembler(interface_mesh, self.area)]
        )

        return

    def get_lsf_coordinates(self):
        return self.lsf_x, self.lsf_y

    def set_lsf(self, lsf):
        self.cut_mesh.get_lsf()[:] = lsf

        self._update_after_design_change()
        return

    def eval_objective(self):

        # Solve the frequency problem using the Galerkin-difference approach
        N = 13
        sigma = -0.1
        solver_type = "IRAM"
        eig_atol = 1e-10
        tol = 1e-14

        lam_gd, self.phi = solve_frequency_problem(
            stiffness_assembler=self.stiffness_assembler,
            mass_assembler=self.mass_assembler,
            sigma=sigma,
            N=N,
            eig_atol=eig_atol,
            tol=tol,
        )

        self.frequency = np.sqrt(lam_gd[3:])

        min_freq = self.frequency[0]
        rho = self.ks_param
        ks = min_freq - np.log(np.sum(np.exp(-rho * (self.frequency - min_freq)))) / rho

        return ks

    def eval_objective_derivative(self):

        # Compute the weighting for each eigenvalue derivative
        min_freq = self.frequency[0]
        rho = self.ks_param
        eta = np.exp(-rho * (self.frequency - min_freq))
        sum = np.sum(eta)
        eta = eta / sum

        # Compute the derivative of the objective wrt the eigenvalues
        dfdlam = 0.5 * eta / self.frequency

        # Derivative wrt the lsf values
        dfdx = np.zeros(self.lsf_x.shape)

        for i, freq in enumerate(self.frequency):
            self.stiffness_assembler.get_dof()[:] = self.phi[:, i + 3]
            self.stiffness_assembler.zero_derivative()
            self.stiffness_assembler.add_functional_derivative()
            dfdx += dfdlam[i] * self.stiffness_assembler.get_dfdx()

            self.mass_assembler.get_dof()[:] = self.phi[:, i + 3]
            self.mass_assembler.zero_derivative()
            self.mass_assembler.add_functional_derivative()
            dfdx -= freq**2 * dfdlam[i] * self.mass_assembler.get_dfdx()

        dfdx *= 2.0

        return dfdx

    def eval_area(self):
        return self.area_assembler.eval_functional()

    def eval_area_derivative(self):
        self.area_assembler.zero_derivative()
        self.area_assembler.add_functional_derivative()
        return np.array(self.area_assembler.get_dfdx())

    def eval_perimeter(self):
        return self.perimeter_assembler.eval_functional()

    def eval_perimeter_derivative(self):
        self.perimeter_assembler.zero_derivative()
        self.perimeter_assembler.add_functional_derivative()
        return np.array(self.perimeter_assembler.get_dfdx())


tree = xcgd.Quadtree()

# Uniformly refine the mesh to have 2**5 = 32 elements along each edge
tree.refine([4])
tree.balance()

opt = QuadtreeOpt(tree, length=3.0)

x0 = 1.5
y0 = 1.5
r0 = 1.0

x, y = opt.get_lsf_coordinates()
lsf = (x - x0) ** 2 + (y - y0) ** 2 - r0**2
opt.set_lsf(lsf)

f0 = opt.eval_objective()
a0 = opt.eval_area()
p0 = opt.eval_perimeter()
dfdx = opt.eval_objective_derivative()
dadx = opt.eval_area_derivative()
dpdx = opt.eval_perimeter_derivative()

dh = 1e-6
pert = np.ones(len(lsf))

analytic = np.dot(dfdx, pert)
a_analytic = np.dot(dadx, pert)
p_analytic = np.dot(dpdx, pert)

opt.set_lsf(lsf + dh * pert)

f1 = opt.eval_objective()
a1 = opt.eval_area()
p1 = opt.eval_perimeter()

fd = (f1 - f0) / dh
a_fd = (a1 - a0) / dh
p_fd = (p1 - p0) / dh

print(f"f0       {f0:20.10e}")
print(f"analytic {analytic:20.10e}")
print(f"fd       {fd:20.10e}")
print(f"rel err  {((analytic - fd) / fd):20.10e}")

print(f"a0       {a0:20.10e}")
print(f"analytic {a_analytic:20.10e}")
print(f"fd       {a_fd:20.10e}")
print(f"rel err  {((a_analytic - a_fd) / a_fd):20.10e}")

print(f"p0       {p0:20.10e}")
print(f"analytic {p_analytic:20.10e}")
print(f"fd       {p_fd:20.10e}")
print(f"rel err  {((p_analytic - p_fd) / p_fd):20.10e}")
