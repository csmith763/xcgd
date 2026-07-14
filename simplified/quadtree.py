import xcgd
import numpy as np

tree = xcgd.Quadtree()

# Uniformly refine the mesh to have 2**5 = 32 elements along each edge
tree.refine([5])
tree.balance()

# Create a random number generator, but keep it repeatable
rng = np.random.default_rng(12345)

lower = -1
upper = 3
refinement = rng.integers(lower, upper, size=tree.size(), dtype=np.int32)
tree.refine(refinement)
tree.balance()

lower = -1
upper = 3
refinement = rng.integers(lower, upper, size=tree.size(), dtype=np.int32)
tree.refine(refinement)
tree.balance()

# Write out the quadtree
tree.to_vtk("test_tree_balanced.vtk")

# Create a quadtree mesh
length = 1.0
mesh = xcgd.QuadtreeMesh(tree, length)

# Set up the physics on the mesh
E, nu, rho = 70.0e3, 0.3, 1.0
elas = xcgd.LinearElasticity2D(E, nu)
assembler = xcgd.Assembler([xcgd.LinearElasticity2DAssembler(mesh, elas)])

# Set up everything
assembler.update()
assembler.eval_jacobian()
jac = assembler.get_jacobian()
