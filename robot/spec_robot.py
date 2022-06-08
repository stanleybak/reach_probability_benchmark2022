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

def set_settings():
    'set settings'

    #Settings.NUM_PROCESSES = 1
    Settings.TIMING_STATS = True
    #Settings.TRY_QUICK_OVERAPPROX = False

    #Settings.CONTRACT_ZONOTOPE_LP = True
    #Settings.CONTRACT_LP_OPTIMIZED = True
    #Settings.CONTRACT_LP_TRACK_WITNESSES = True

    #Settings.OVERAPPROX_BOTH_BOUNDS = False

    #Settings.BRANCH_MODE = Settings.BRANCH_EXACT
    Settings.RESULT_SAVE_STARS = False

def main():
    """main entry point"""

    plot = True
    onnx_filename = "robot.onnx"
    set_settings()

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

    # spec, state is near the origin and probability is above a threshold
    log_prob = 0.5
    rad = 0.2

    # rad 0.1, log_prob = 0.07
    # rad 0.2, log_prob 0.2
    
    mat = np.array([[0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [-1, 0, 0, 0, 0]], dtype=float)
    rhs = np.array([rad, rad, rad, rad, -log_prob], dtype=float)
    spec = Specification(mat, rhs)

    res = enumerate_network(init_box, network, spec)

    print(f"result in {res.total_secs} was: {res.result_str}")

    print(f"cinput: {res.cinput}")
    print(f"coutput: {res.coutput}")
        
if __name__ == "__main__":
    main()
