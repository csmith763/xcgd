import numpy as np
import xcgd as xd
import amigo as am
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix


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


Lx = 1.0

nx = 128
ny = 32
delta = Lx / nx
Ly = (ny / nx) * Lx
mesh = xd.CartesianMesh(nx, ny, delta)

E, nu = 70.0e3, 0.3
phys = xd.LinearElasticity2D(E, nu)
elasticity_assembler = xd.LinearElasticity2DAssembler(mesh, phys)
assembler = xd.Assembler([elasticity_assembler])

# Update the CSR nonzero pattern and DOF data. This
# is required after any connectivity change
assembler.update()

# # Evaluate the residual and the Jacobian
assembler.eval_residual()
assembler.eval_jacobian()

# Retrieve the Jacobian we just computed
jac = assembler.get_jacobian()

# Assemble a CSR matrix
mat = csr_matrix((jac.data, jac.cols, jac.rowp), shape=(jac.nrows, jac.nrows))

# Copy the data that we're about to modify
data = np.copy(jac.data)

rhs = np.zeros(2 * (nx + 1) * (ny + 1))
rhs[1::2] = 1.0

# Rows to zero
zero_nodes = np.arange(0, (nx + 1) * (ny + 1), nx + 1)
zero_dof = np.zeros(2 * len(zero_nodes), dtype=int)
zero_dof[0::2] = 2 * zero_nodes
zero_dof[1::2] = 2 * zero_nodes + 1

zero_rows_and_columns(zero_dof, jac.nrows, jac.rowp, jac.cols, data)
rhs[zero_dof] = 0.0

# Use the modified data and right-hand-side
mat = am.CSRMat(jac.nrows, jac.nrows, jac.rowp, jac.cols, data)

ldl = am.SparseLDL(
    mat, solver_type=am.SolverType.LDL, ustab=0.04, order=am.OrderingType.DEFAULT
)
ldl.factor()

X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))

ldl.solve(rhs)

# Get the dof and set them with the solution
dof = assembler.get_dof()
dof[:] = rhs

U = rhs[::2].reshape((ny + 1, nx + 1))
V = rhs[1::2].reshape((ny + 1, nx + 1))

fig, ax = plt.subplots()
ax.contourf(X + 0.05 * U, Y + 0.05 * V, V)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.tight_layout(pad=0)

plt.show()
