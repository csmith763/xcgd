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


tree = xcgd.Quadtree()

# Uniformly refine the mesh to have 2**5 = 32 elements along each edge
tree.refine([4])
tree.balance()

# Duplicate the quadtree
source = tree.duplicate()

source.to_vtk("source.vtk")

for i in range(3):
    tree.refine()
    tree.balance()  # NOTE: need to balance each time after refine is called

# Write out the quadtree
tree.to_vtk("test_tree_balanced.vtk")
length = 3.0

# Create the LSF mesh
lsf_mesh = xcgd.QuadtreeMesh(source, length)

# Create a quadtree mesh
mesh = xcgd.QuadtreeMesh(tree, length)

# Get the node locations and specify the level set function
X = np.array(lsf_mesh.get_node_locations())

cut_mesh = xcgd.QuadtreeCutMesh(mesh, lsf_mesh)
lsf = cut_mesh.get_lsf()

x0 = 1.5
y0 = 1.5
r0 = 1.0
lsf[:] = (X[::2] - x0) ** 2 + (X[1::2] - y0) ** 2 - r0**2

cut_mesh.update()
interface_elems = cut_mesh.get_interface_elements()

interior_elems = cut_mesh.get_interior_elements()

refinement = np.zeros(tree.size(), dtype=np.int32)
refinement[interface_elems] = 2
refinement[interior_elems] = 1
tree.refine(refinement)
tree.balance()

tree.to_vtk("refined_quadtree.vtk")

mesh.update()
cut_mesh.update()

interior_mesh = cut_mesh.create_interior_mesh()


E, nu, rho = 1.0, 0.3, 1.0
elas = xcgd.LinearElasticity2D(E, nu)
mass = xcgd.ElasticityMass2D(rho)

stiffness_assembler = xcgd.Assembler(
    [xcgd.LinearElasticity2DAssembler(interior_mesh, elas)]
)
mass_assembler = xcgd.Assembler([xcgd.ElasticityMass2DAssembler(interior_mesh, mass)])

# Solve the frequency problem using the Galerkin-difference approach
N = 40
sigma = -0.1
solver_type = "IRAM"
eig_atol = 1e-10
tol = 1e-14

lam_gd, _ = solve_frequency_problem(
    stiffness_assembler=stiffness_assembler,
    mass_assembler=mass_assembler,
    sigma=sigma,
    N=N,
    eig_atol=eig_atol,
    tol=tol,
)

frequencies = np.sqrt(np.abs(lam_gd))

for i, freq in enumerate(frequencies[3:]):
    print(f"{i:2d}: {freq:20.12f}")


# # Set up the physics on the mesh
# E, nu, rho = 70.0e3, 0.3, 1.0
# elas = xcgd.LinearElasticity2D(E, nu)
# assembler = xcgd.Assembler([xcgd.LinearElasticity2DAssembler(mesh, elas)])

# # Set up everything
# assembler.update()
# assembler.eval_jacobian()
# jac = assembler.get_jacobian()
