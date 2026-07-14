import numpy as np
import xcgd as xd
import amigo as am
import matplotlib.pylab as plt
import time


def zero_rows_and_columns(zero_dof, nrows, rowp, cols, data):
    zero_dof = np.asarray(zero_dof, dtype=int)

    # Zero selected rows
    for dof in zero_dof:
        data[rowp[dof] : rowp[dof + 1]] = 0.0

    # Zero selected columns
    col_mask = np.isin(cols, zero_dof)
    data[col_mask] = 0.0

    # Restore diagonal entries
    for dof in zero_dof:
        start = rowp[dof]
        end = rowp[dof + 1]

        diag = np.where(cols[start:end] == dof)[0]
        if len(diag) == 0:
            raise RuntimeError(f"Missing diagonal entry in row {dof}")

        data[start + diag[0]] = 1.0


def apply_boundary_conditions(nx, ny, csr):
    # Copy the data that we're about to modify
    data = np.copy(csr.data)

    rhs = np.zeros(2 * (nx + 1) * (ny + 1))
    rhs[1::2] = 1.0

    # Rows to zero
    zero_nodes = np.arange(0, (nx + 1) * (ny + 1), nx + 1)
    zero_dof = np.zeros(2 * len(zero_nodes), dtype=int)
    zero_dof[0::2] = 2 * zero_nodes
    zero_dof[1::2] = 2 * zero_nodes + 1

    zero_rows_and_columns(zero_dof, csr.nrows, csr.rowp, csr.cols, data)
    rhs[zero_dof] = 0.0

    mat = am.CSRMat(csr.nrows, csr.nrows, csr.rowp, csr.cols, data)

    return mat, rhs


Lx = 3.0

nx = 256
ny = 256
delta = Lx / nx
Ly = (ny / nx) * Lx
mesh = xd.CartesianMesh(nx, ny, delta)

# Set up the problem radius
radius = 1.0

E, nu, rho = 70.0e3, 0.3, 1.0
elas = xd.LinearElasticity2D(E, nu)
mass = xd.ElasticityMass2D(rho)

# Get the x/y coordinates
X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))

# Compute a level set function
lsf = (X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2 - radius**2

# Set the level set function
cut_mesh = xd.CartesianCutMesh(mesh)
cut_mesh.get_lsf()[:] = lsf.flatten()

t0 = time.perf_counter()
cut_mesh.update()
t1 = time.perf_counter()
cut_mesh.update_derivatives()
t2 = time.perf_counter()

print("update time: ", t1 - t0)
print("derivative time: ", t2 - t1)

interior_mesh = cut_mesh.create_interior_mesh()

interior_stiffness_assembler = xd.Assembler(
    [xd.LinearElasticity2DAssembler(interior_mesh, elas)]
)
interior_mass_assembler = xd.Assembler(
    [xd.ElasticityMass2DAssembler(interior_mesh, mass)]
)
interior_stiffness_assembler.update()
interior_mass_assembler.update()

interior_stiffness_assembler.eval_jacobian()
interior_mass_assembler.eval_jacobian()

kcsr = interior_stiffness_assembler.get_jacobian()
mcsr = interior_mass_assembler.get_jacobian()

# Use the modified data and right-hand-side
# kmat = am.CSRMat(kcsr.nrows, kcsr.nrows, kcsr.rowp, kcsr.cols, kcsr.data)
# mmat = am.CSRMat(mcsr.nrows, mcsr.nrows, mcsr.rowp, mcsr.cols, mcsr.data)

# mat, rhs = apply_boundary_conditions(nx, ny, kcsr)

# ldl = am.SparseLDL(
#     mat, solver_type=am.SolverType.LDL, ustab=0.04, order=am.OrderingType.DEFAULT
# )
# ldl.factor()

# ldl.solve(rhs)

# # Get the dof and set them with the solution
# dof = stiffness_assembler.get_dof()
# dof[:] = rhs

# U = rhs[::2].reshape((ny + 1, nx + 1))
# V = rhs[1::2].reshape((ny + 1, nx + 1))

# fig, ax = plt.subplots()
# ax.contourf(X + 0.05 * U, Y + 0.05 * V, V)
# ax.set_aspect("equal")
# ax.axis("off")
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# fig.tight_layout(pad=0)

# plt.show()
