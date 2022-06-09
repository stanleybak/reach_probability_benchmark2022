"""
Reachset for vanderpol density
"""

import sys
from math import pi

import numpy as np
from numpy import deg2rad

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

def get_scaling():
    """get network input / output scaling"""

    rv = {
        'in_means':
            np.array([ 5.80105554e+02,  8.75405736e-02, -2.27912875e-05, -8.76908046e-02,
            -7.46272425e-01, -1.92803375e-04,  3.34716638e-05,  2.89071084e-05,
            9.25385983e-05,  7.56612315e-06,  7.76800018e-05,  1.17481262e+03,
            5.04494635e-01,  0.00000000e+00]),
        'in_stds':
            np.array([1.14912446e+01, 7.24599774e-03, 5.80582688e-03, 7.19496912e-03,
            1.43647092e-01, 5.79842891e-03, 5.81868057e-03, 5.71455786e-03,
            5.82288037e-03, 5.75686386e-03, 5.84082689e-03, 1.46177912e+01,
            2.88088940e-01, 1.00000000e+00]),
        'out_means':
            np.array([ 0.00000000e+00,  5.72248962e+02,  1.87254448e-01, -6.63293962e-04,
            -5.94605300e-02, -2.29877173e-01, -4.05609419e-02,  3.78253769e-03,
            2.60706149e-01, -2.43826457e-03,  8.21500599e+02, -1.53932647e+01,
            6.39446683e+02,  6.66933037e+00]),
        'out_stds':
            np.array([1.06752443e+02, 2.11187233e+01, 5.78059807e-02, 1.54801777e-03,
            1.20764860e-02, 3.33593232e-01, 2.40607286e-02, 1.75967886e-02,
            1.85259960e-01, 4.45550252e-03, 5.22783675e+02, 1.95354034e+01,
            2.79991020e+02, 2.23535258e+00])
        }

    return rv

def main():
    """main entry point"""

    plot = True
    onnx_filename = "gcas.onnx"
    set_settings()

    #onnx_model = onnx.load(onnx_filename)
    #onnx.checker.check_model(onnx_model, full_check=True)
    #print("check passed!")
    #exit(1)

    scaling_dict = get_scaling()

    #trange = (0, 106 * 1/30)
    trange = (105 * 1/30, 105 * 1/30 + 1e-4)

    if False:

        vt = 540
        alpha = deg2rad(2.1215)
        beta = 0

        phi = -pi/8           # Roll angle from wings level (rad)
        #theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
        theta = 0         # Pitch angle from nose level (rad)
        psi = 0   # Yaw angle from North (rad)

        P = 0
        Q = 0
        R = 0

        Pn  = 0
        Pe = 0

        alt = 1200
        power = 8

        pt = [vt, alpha, beta, phi, theta, psi, P, Q, R, Pn, Pe, alt, power]

        tuples_list = [(x, x) for x in pt] + [trange]
    else:
        # yue paper init set
        tuples_list = [(560.0040, 599.9999),
            (0.0750, 0.1000),
            (-0.0100, 0.0100),
            (-0.1000, -0.0750),
            (-0.9996, -0.5000),
            (-0.0100, 0.0100),
            (-0.0100, 0.0100),
            (-0.0100, 0.0100),
            (-0.0100, 0.0100),
            (-0.0100, 0.0100),
            (-0.0100, 0.0100),
            (1150.0081, 1199.9975),
            (0.0002, 0.9998),
            trange]

    # scale input
    for i, tup in enumerate(tuples_list):
        lb = (tup[0] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]
        ub = (tup[1] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]

        tuples_list[i] = (lb, ub)
    
    init_box = np.array(tuples_list, dtype=float)
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
    log_prob = -9999

    # rad 0.1, log_prob = 0.07
    # rad 0.2, log_prob 0.2
    alt_index = 12
    prob_index = 0
    min_alt = 0 # 42.9
    
    row1 = [1 if i == alt_index else 0 for i in range(14)]
    row2 = [-1 if i == prob_index else 0 for i in range(14)]
    
    mat = np.array([row1, row2], dtype=float)
    rhs = np.array([min_alt, -log_prob], dtype=float)

    # apply output scaling to rhs
    rhs_indices = [alt_index, prob_index] ##### HERE !!!!!!!
    
    for rhs_index, (i, x) in enumerate(zip(rhs_indices, rhs)):
        newx = (x - scaling_dict['out_means'][i]) / scaling_dict['out_stds'][i]
        rhs[rhs_index] = newx

        print(f"rhs {x} -> {newx}")
        
    spec = Specification(mat, rhs)
    res = enumerate_network(init_box, network, spec)

    print(f"result in {res.total_secs} was: {res.result_str}")

    print(f"rhs: {rhs}")

    ucinput = []

    if res.cinput:
        for i, x in enumerate(res.cinput):
            u = x * scaling_dict['in_stds'][i] + scaling_dict['in_means'][i]
            ucinput.append(u)

    print(f"\nunscaled cinput: {ucinput}")
    
    print(f"\ncinput: {res.cinput}")
    print(f"\ncoutput: {res.coutput}")

    ucoutput = []

    if res.coutput:
        for i, x in enumerate(res.coutput):
            u = x * scaling_dict['out_stds'][i] + scaling_dict['out_means'][i]
            ucoutput.append(u)

    print(f"\nunscaled coutput: {ucoutput}")
    
        
if __name__ == "__main__":
    main()
