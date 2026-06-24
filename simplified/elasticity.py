import numpy as np
import xcgd as xd
import amigo as am
import matplotlib.pylab as plt


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


Lx = 1.0

nx = 128
ny = 32
delta = Lx / nx
Ly = (ny / nx) * Lx
mesh = xd.CartesianMesh(nx, ny, delta)

E, nu, rho = 70.0e3, 0.3, 1.0
elas = xd.LinearElasticity2D(E, nu)
mass = xd.ElasticityMass2D(rho)
stiffness_assembler = xd.Assembler([xd.LinearElasticity2DAssembler(mesh, elas)])
mass_assembler = xd.Assembler([xd.ElasticityMass2DAssembler(mesh, mass)])

# Update the CSR nonzero pattern and DOF data. This
# is required after any connectivity change
stiffness_assembler.update()
mass_assembler.update()

# # Evaluate the residual and the Jacobian
stiffness_assembler.eval_jacobian()
mass_assembler.eval_jacobian()

# Retrieve the Jacobian we just computed
kcsr = stiffness_assembler.get_jacobian()
mcsr = mass_assembler.get_jacobian()

# Get the x/y coordinates
X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))

# Compute a level set function
lsf = 1 - 10 * (X - 0.5) ** 2 + 4 * (Y - 0.125) ** 2

# Set the level set function
cut_mesh = xd.CartesianCutMesh(mesh)
cut_mesh.get_lsf()[:] = lsf.flatten()

cut_mesh.update()

# Use the modified data and right-hand-side
# kmat = am.CSRMat(kcsr.nrows, kcsr.nrows, kcsr.rowp, kcsr.cols, kcsr.data)
# mmat = am.CSRMat(mcsr.nrows, mcsr.nrows, mcsr.rowp, mcsr.cols, mcsr.data)

mat, rhs = apply_boundary_conditions(nx, ny, kcsr)

ldl = am.SparseLDL(
    mat, solver_type=am.SolverType.LDL, ustab=0.04, order=am.OrderingType.DEFAULT
)
ldl.factor()

ldl.solve(rhs)

# Get the dof and set them with the solution
dof = stiffness_assembler.get_dof()
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
