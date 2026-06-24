import numpy as np
import xcgd as xd
import amigo as am
import matplotlib.pylab as plt

Lx = 1.0

nx = 128
ny = 32
delta = Lx / nx
Ly = (ny / nx) * Lx

r = 0.05

phys = xd.Helmholtz(r)
mesh = xd.CartesianMesh(nx, ny, delta)
helmholtz_assembler = xd.HelmholtzAssembler(mesh, phys)
assembler = xd.Assembler([helmholtz_assembler])

# Update the CSR nonzero pattern and DOF data. This
# is required after any connectivity change
assembler.update()

# Evaluate the residual and the Jacobian
assembler.eval_residual()
assembler.eval_jacobian()

jac = assembler.get_jacobian()
mat = am.CSRMat(jac.nrows, jac.nrows, jac.rowp, jac.cols, jac.data)

ldl = am.SparseLDL(
    mat, solver_type=am.SolverType.LDL, ustab=0.04, order=am.OrderingType.DEFAULT
)

ldl.factor()

X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))

rhs = 1.0 - np.exp(-((X - 0.5) ** 2 + 4.0 * (Y - 0.125) ** 2))
rhs = rhs.flatten()

ldl.solve(rhs)

# Get the dof and set them with the solution
dof = assembler.get_dof()
dof[:] = rhs

F = rhs.reshape((ny + 1, nx + 1))
plt.contourf(X, Y, F)
plt.show()
