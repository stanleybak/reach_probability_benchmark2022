"""
Reachset for vanderpol density
"""

import sys
from math import pi

import numpy as np

import onnx
import matplotlib.pyplot as plt

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

def set_settings(plot):
    'set settings'

    #Settings.NUM_PROCESSES = 1
    Settings.TIMING_STATS = True
    Settings.TRY_QUICK_OVERAPPROX = False

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_EXACT
    Settings.RESULT_SAVE_STARS = plot

def main():
    """main entry point"""

    plot = True
    onnx_filename = "model499000.onnx"
    set_settings(plot)

    #onnx_model = onnx.load(onnx_filename)
    #onnx.checker.check_model(onnx_model, full_check=True)
    #print("check passed!")
    #exit(1)

    tmax = 0.05 * 50
    init_box = np.array([(-1.8, -1.2), (-1.8, -1.2), (0, pi/2), (1.0, 1.5), (0, tmax)], dtype=float)
    #init_box = np.array([[0.9, 1.1], [0.9, 1.1], [0, 5.0]], dtype=float)
    #init_box = np.array([[-2.5, 2.5], [-2.5, 2.5], [0, 5.0]], dtype=float)

    try:
        network = load_onnx_network_optimized(onnx_filename)
        print("loaded optimized!")
    except:
        print("optimized network load failed, using normal load")
        # cannot do optimized load due to unsupported layers
        network = load_onnx_network(onnx_filename)

    res = enumerate_network(init_box, network)

    print(f"result in {res.total_secs} was: {res.result_str}")
    print(f"num stars: {res.total_stars} ({len(res.stars)})")

    if plot:
        for i, star in enumerate(res.stars):

            if i % 1000 == 0:
                print(f"plotting ({round(100 * i/len(res.stars), 2)}%)...", flush=True)

            verts = star.verts(xdim=1, ydim=2)

            plt.fill(*zip(*verts), 'b-')

        xs = [init_box[0][0], init_box[0][0], init_box[0][1], init_box[0][1], init_box[0][0]]
        ys = [init_box[1][0], init_box[1][1], init_box[1][1], init_box[1][0], init_box[1][0]]
        plt.plot(xs, ys, 'r-', lw=3, label='Init set')

        plt.legend()
        plt.title("Computed reach set from nnenum")
        plt.savefig('reachset_computed.png')
    
if __name__ == "__main__":
    main()
