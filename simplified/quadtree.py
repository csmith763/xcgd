import xcgd
import numpy as np

tree = xcgd.Quadtree()
print("quadtree", flush=True)

# Uniformly refine the mesh to have 2**5 = 32 elements along each edge
tree.refine([2])
print("refine", flush=True)
tree.balance()
print("balance", flush=True)

# Duplicate the quadtree
source = tree.duplicate()

# Write out the quadtree
tree.to_vtk("test_tree_balanced.vtk")
length = 1.0

# Create the LSF mesh
lsf_mesh = xcgd.QuadtreeMesh(source, length)

# Create a quadtree mesh
mesh = xcgd.QuadtreeMesh(tree, length)

# Get the node locations and specify the level set function
X = np.array(lsf_mesh.get_node_locations())

cut_mesh = xcgd.QuadtreeCutMesh(mesh, lsf_mesh)
lsf = cut_mesh.get_lsf()

x0 = 0.5
y0 = 0.5
r0 = 1.0 / np.sqrt(5.0)
lsf[:] = (X[::2] - x0) ** 2 + (X[1::2] - y0) ** 2 - r0**2

cut_mesh.update()
interface_elems = cut_mesh.get_interface_elements()

refinement = np.zeros(tree.size(), dtype=np.int32)
refinement[interface_elems] = 2
tree.refine(refinement)
tree.balance()

tree.to_vtk("refined_quadtree.vtk")

mesh.update()
cut_mesh.update()

# # Set up the physics on the mesh
# E, nu, rho = 70.0e3, 0.3, 1.0
# elas = xcgd.LinearElasticity2D(E, nu)
# assembler = xcgd.Assembler([xcgd.LinearElasticity2DAssembler(mesh, elas)])

# # Set up everything
# assembler.update()
# assembler.eval_jacobian()
# jac = assembler.get_jacobian()
