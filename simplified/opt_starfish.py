from flume.base_classes.system import System
from flume.interfaces.paropt_interface import FlumeParOptInterface
import numpy as np
from simplified.xcgd_flume_classes import (
    Starfish,
    XCGDAnalysis,
    XCGDArea,
    XCGDMinFrequency,
    XCGDPerimeter,
)
import xcgd
from math import pi
from paropt import ParOpt
import matplotlib.pyplot as plt
from icecream import ic
import shutil
import os


class StarfishCallback:

    def __init__(self, starfish: Starfish):

        # Store the starfish object
        self.starfish = starfish

    def __call__(self, x, it_num):

        if it_num % 5 == 0:
            # Plot the LSF contour
            fig = self.starfish.plot_lsf()

            savename = f"output/contours/lsf_contour_it_{it_num}.png"
            fig.savefig(savename, bbox_inches="tight", dpi=200)

            plt.close(fig)

        return


if __name__ == "__main__":

    # Construct the source tree for the LSF function
    tree = xcgd.Quadtree()

    # Uniformly refine the mesh to have 2**5 = 32 elements along each edge
    tree.refine([5])
    tree.balance()

    tree.to_vtk("output/background_mesh.vtk")

    if os.path.exists("output/contours"):
        shutil.rmtree("output/contours")

    os.makedirs("output/contours")

    # Construct the XCGD analysis object
    mesh_length = 3.0
    xcgd_analysis = XCGDAnalysis(lsf_tree=tree, mesh_length=mesh_length)

    # Construct the design variable object
    ndvs = 5
    starfish = Starfish(
        xcgd_analysis=xcgd_analysis, obj_name="starfish", sub_analyses=[], ndvs=ndvs
    )

    c0 = np.random.uniform(low=0.2, high=0.3, size=ndvs)

    starfish.set_var_values(variables={"coeffs": c0})

    # Construct the XCGDFrequencyAnalysisObject
    ks_param = 50.0
    xcgd_freq = XCGDMinFrequency(
        xcgd_analysis=xcgd_analysis,
        obj_name="XCGD_Frequency",
        sub_analyses=[starfish],
        ks_param=ks_param,
    )

    # Construct the XCGDArea object
    area = XCGDArea(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Area", sub_analyses=[starfish]
    )

    # Construct the XCGDPerimeter object
    perimeter = XCGDPerimeter(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Perimeter", sub_analyses=[starfish]
    )

    # Define the System
    system = System(
        sys_name="XCGD_Starfish_Opt",
        top_level_analysis_list=[xcgd_freq, area, perimeter],
        log_name="flume.log",
        log_prefix="output",
    )

    # Define the design variables for the system
    coeffs_lb = -0.5
    coeffs_ub = 0.5

    system.declare_design_vars(
        global_var_name={"starfish.coeffs": {"lb": coeffs_lb, "ub": coeffs_ub}}
    )

    # Declare the objective for the system
    system.declare_objective(
        global_obj_name="XCGD_Frequency.omega_ks_min", obj_scale=-1.0
    )

    # Declare the constraints for the system (assuming area = pi is the equality constraint)
    alpha = 1.2
    area_val = pi
    perim_val = alpha * 2 * pi

    ic(area_val)
    ic(perim_val)

    system.declare_constraints(
        global_con_name={
            "XCGD_Area.area": {"rhs": area_val, "direction": "both"},
            "XCGD_Perimeter.perimeter": {"rhs": perim_val, "direction": "both"},
        }
    )

    # Setup the FlumeParOptInterface
    starfish_callback = StarfishCallback(starfish=starfish)

    interface = FlumeParOptInterface(flume_sys=system, callback=starfish_callback)

    # Create the ParOpt problem and get the options
    paroptprob = interface.construct_paropt_problem()

    maxit = 200
    options = interface.get_paropt_default_options(
        output_prefix="output", algorithm="mma", maxit=maxit
    )
    options["mma_move_limit"] = 0.25
    options["mma_init_asymptote_offset"] = 0.5
    options["mma_asymptote_relax"] = 1.5

    # for i in range(2):
    #     paroptprob.checkGradients(1e-6)
    #     exit()

    # exit()
    # Perform the optimization
    opt = ParOpt.Optimizer(paroptprob, options)
    opt.optimize()

    # Extract the optimized point
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    # Write the optimized point to the json file
    x_opt = np.array(x)

    ic(x_opt)
