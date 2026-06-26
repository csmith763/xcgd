import numpy as np
import xcgd as xd
import amigo as am
import matplotlib.pylab as plt
from elasticity import apply_boundary_conditions
from scipy.sparse import csr_matrix
from eigd import IRAM, make_operator
from icecream import ic
from flume_topology.analyses.topo_analysis import TopoAnalysis
from flume_topology.analyses.frequency_analysis import NaturalFrequencyAnalysis
from flume_topology.utils.mesh_utils import create_beam_domain
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


def create_beam_domain(Lx: float, Ly: float, nx: int, ny: int):
    """
    Creates a beam domain using the inputs for the lengths in the x- and y-directions and the number of elements in the x- and y-directions.
    """

    m = nx
    n = ny

    # make sure m and n are odd, as the number of nodes on each edge must be even for node-based symmetry
    if n % 2 == 0:
        n -= 1
    if m % 2 == 0:
        m -= 1

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, Ly, n + 1)
    x = np.linspace(0, Lx, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j * (m + 1), 0] = x[i]
            X[i + j * (m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j * m, 0] = nodes[j, i]
            conn[i + j * m, 1] = nodes[j, i + 1]
            conn[i + j * m, 2] = nodes[j + 1, i + 1]
            conn[i + j * m, 3] = nodes[j + 1, i]

    # Return an empty list of non-design nodes
    non_design_nodes = []

    # Return empty dictionaries for the forces and bcs
    bcs = {}
    forces = {}

    # Set the dvmap and ndvs to None
    dvmap = ndvs = None

    return conn, X, bcs, forces, non_design_nodes, dvmap, ndvs


def plot_level_set(X, Y, lsf, ax=None):
    """
    Visualizes the level set function over the mesh.

    The filled contours show the value of the level set function, the solid
    black contour marks the zero level set (the material boundary), and the
    shaded overlay highlights the solid (interior) region where ``lsf <= 0``.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinate arrays of shape ``(ny + 1, nx + 1)``.
    lsf : np.ndarray
        Level set values. May be flat (length ``(nx + 1) * (ny + 1)``) or
        already shaped like ``X``; it is reshaped to match ``X`` as needed.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not provided.

    Returns
    -------
    fig, ax : the matplotlib figure and axes containing the plot.
    """

    # Reshape the level set to match the coordinate grid if needed
    phi = np.asarray(lsf).reshape(X.shape)

    # Create the axes if one was not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Filled contours of the level set field
    cf = ax.contourf(X, Y, phi, levels=50, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label=r"$\phi$")

    # # Highlight the solid region (phi <= 0)
    # ax.contourf(
    #     X, Y, phi, levels=[phi.min(), 0.0], colors=["#999999"], alpha=0.4
    # )

    # Mark the zero level set (material boundary)
    ax.contour(X, Y, phi, levels=[0.0], colors="black", linewidths=1.5)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Level Set Function")

    return fig, ax


def plot_frequencies_and_rel_error(lam_gd_nz, lam_nz):
    """
    Constructs a figure where the first subplot is a bar plot showing the actual values for the frequencies contained in the two arrays (one from Galerkin difference, the other from the conventional finite element method). The second subplot shows the relative error for each mode number.
    """

    # Create the figure
    fig, axd = plt.subplot_mosaic(mosaic=[["A"], ["B"]], sharex=True)

    # Plot the values of the natural frequencies as a bar chart
    width = 0.4
    fontsize = 14
    ticksize = 10

    mode_nums = np.arange(lam_gd_nz.size)
    axd["A"].bar(
        mode_nums - width / 2,
        height=lam_gd_nz,
        width=width,
        color="#f77659",
        label="GD",
    )

    axd["A"].bar(
        mode_nums + width / 2, height=lam_nz, width=width, color="#428fd7", label="Q4"
    )
    axd["A"].set_ylabel(r"$\omega$ (rad/s)", fontweight="normal", fontsize=fontsize)
    axd["A"].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    axd["A"].tick_params(axis="both", which="both", labelsize=ticksize)

    offset = axd["A"].yaxis.get_offset_text()
    offset.set_fontsize(ticksize)
    offset.set_x(-0.05)

    # Compute the relative error for each frequency
    rel_err = np.abs(lam_nz - lam_gd_nz) / lam_nz

    # Plot the relative error
    axd["B"].bar(mode_nums, rel_err, width=0.8, color="#E38A24")
    axd["B"].set_yscale("log")
    axd["B"].set_ylim([1e-4, 1e-1])
    axd["B"].set_xlabel("Frequency Number", fontweight="normal", fontsize=10)
    axd["B"].set_ylabel("Rel. Error", fontweight="normal", fontsize=10)
    axd["B"].tick_params(axis="both", which="both", labelsize=ticksize)

    return fig, axd


if __name__ == "__main__":

    # Set the plot style
    plt.style.use(niceplots.get_style())
    plt.rcParams["font.family"] = "helvetica"

    Lx = 3.0

    # nx = 256
    # ny = 256
    # nx = 128
    # ny = 128
    # nx = 64
    # ny = 64
    # nx = 32
    # ny = 32

    nx = 8
    ny = 8

    delta = Lx / nx
    Ly = (ny / nx) * Lx
    mesh = xd.CartesianMesh(nx, ny, delta)

    # Set up the problem radius
    radius = 1.0

    E, nu, rho = 1.0, 0.3, 1.0
    elas = xd.LinearElasticity2D(E, nu)
    mass = xd.ElasticityMass2D(rho)

    # Get the x/y coordinates
    X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))

    # Compute a level set function
    lsf = (X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2 - radius**2

    # Visualize the level set function
    plot_level_set(X, Y, lsf)
    plt.show()

    # Set the level set function
    cut_mesh = xd.CartesianCutMesh(mesh)
    cut_mesh.get_lsf()[:] = lsf.flatten()

    cut_mesh.update()

    X = cut_mesh.get_node_locations()
    ic(X.shape)

    exit()

    interior_mesh = cut_mesh.create_interior_mesh()

    stiffness_assembler = xd.Assembler(
        [xd.LinearElasticity2DAssembler(interior_mesh, elas)]
    )
    mass_assembler = xd.Assembler([xd.ElasticityMass2DAssembler(interior_mesh, mass)])

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

    exit()

    # Construct the Cartesian mesh
    Lx = 1.0
    nx = 128
    ny = 32
    delta = Lx / nx
    Ly = (ny / nx) * Lx

    mesh = xd.CartesianMesh(nx, ny, delta)

    # Define the material parameters
    E, nu, rho = 70.0e3, 0.3, 1.0

    # Define the physics for the linear elasticity and mass
    elas = xd.LinearElasticity2D(E, nu)
    mass = xd.ElasticityMass2D(rho)

    # Define the assemblers for the stiffness and mass matrices
    stiffness_assembler = xd.Assembler([xd.LinearElasticity2DAssembler(mesh, elas)])
    mass_assembler = xd.Assembler([xd.ElasticityMass2DAssembler(mesh, mass)])

    # Solve the frequency problem using the Galerkin-difference approach
    N = 20
    sigma = 0.0
    solver_type = "IRAM"
    eig_atol = 1e-5
    tol = 1e-14

    lam_gd, _ = solve_frequency_problem(
        stiffness_assembler=stiffness_assembler,
        mass_assembler=mass_assembler,
        sigma=0.0,
        N=N,
        eig_atol=eig_atol,
        tol=tol,
    )

    # Construct the mesh again in a format compatible with the TopoAnalysis class
    conn, X, bcs, forces, non_design_nodes, dvmap, ndvs = create_beam_domain(
        Lx=Lx, Ly=Ly, nx=nx, ny=ny
    )

    # Construct the TopoAnalysis object
    topo = TopoAnalysis(
        conn=conn,
        X=X,
        bcs=bcs,
        forces=forces,
        density=rho,
        E=E,
        nu=nu,
        p=1.0,
        kappaK=0.0,
        kappaM=0.0,
        dvmap=dvmap,
        num_design_vars=ndvs,
        non_design_nodes=non_design_nodes,
    )

    # Construct the NaturalFrequencyAnalysis object
    freq = NaturalFrequencyAnalysis(
        topo_analysis=topo,
        obj_name="freq_analysis",
        sub_analyses=[],
        N=N,
        solver_type=solver_type,
        sigma=sigma,
        adjoint_method="sibk",
        tol=tol,
        eig_atol=eig_atol,
    )

    # Set the elemental densities all to 1
    rhoE = np.ones(topo.nelems)
    freq.set_var_values({"rhoE": rhoE})

    # Perform the frequency analysis
    freq.analyze()

    # Extract the natural frequencies
    lam = freq.outputs["lam"].value

    # Remove the rigid body modes
    gd_mask = lam_gd > 1e-6
    lam_gd_nz = lam_gd[gd_mask]

    mask = lam > 1e-6
    lam_nz = lam[mask]

    # Compute the relative error and plot
    plot_frequencies_and_rel_error(lam_gd_nz=lam_gd_nz, lam_nz=lam_nz)

    niceplots.save_figs(plt.gcf(), name="GD_Q4_Frequency_Comparison", formats=["pdf"])

    plt.show()
