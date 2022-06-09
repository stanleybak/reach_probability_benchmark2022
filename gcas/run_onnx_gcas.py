"""
Stanley Bak

Test Vanderpol onnx output
"""

from collections import namedtuple
import math

import onnxruntime as ort
import numpy as np
from numpy import deg2rad

import matplotlib.pyplot as plt

def run_network(sess, x, stdout=False):
    'run the network and return the output'

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, len(x))

    
    outputs = sess.run(None, {'input': in_array})

    if stdout:
        print(f"stdout: {in_array} -> {outputs[0][0]}")

    return outputs[0][0]

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

    sess = ort.InferenceSession("gcas.onnx")
    #sess = ort.InferenceSession("model0.onnx")

    scaling_dict = get_scaling()

    vt = 540
    alpha = deg2rad(2.1215)
    beta = 0

    phi = -0.0877 # -math.pi/8           # Roll angle from wings level (rad)
    #theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)

    P = 0
    Q = 0
    R = 0

    Pn  = 0
    Pe = 0
    
    alt = 1200
    power = 0.5045 #8

    pt = [vt, alpha, beta, phi, theta, psi, P, Q, R, Pn, Pe, alt, power]
    num_inputs = len(pt)

    state = np.array([pt], dtype=float)

    dt = 1/30
    steps = 106

    statenames = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']

    for output_var in range(num_inputs):
        xs = []
        ys = []
    
        for t in np.arange(0, dt * steps, dt):

            unscaled_input = [*pt, t]
            scaled_input = []

            for i in range(len(unscaled_input)):
                #x = (unscaled_input[i] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]
                x = (unscaled_input[i] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]
                scaled_input.append(x)

            print()
            print(f"unscaled_input: {unscaled_input}")
            print(f"scaled_input: {scaled_input}")

            res = run_network(sess, scaled_input, stdout=False)

            xs.append(t)

            y = []

            for i in range(len(unscaled_input)):
                unscaled = res[i] * scaling_dict['out_stds'][i] + scaling_dict['out_means'][i]
                
                y.append(unscaled)
                
            ys.append(y[output_var + 1]) # +1 to get rid of probability
            print(f"output: {res}")
            print(f"unscaled: {y}")
            #exit(1)

            #sys_xs.append(state[0, 0])
            #sys_ys.append(state[0, 1])

            #print(f"state ({state.shape}): {state}")

            #prev_state = state.copy()
            #state = sys.get_next_state(state, 0)
            #print(f"from {prev_state} to {state}")

        #plt.plot(sys_xs, sys_ys, 'r-', lw=0.3 if first else 0.1, label='actual' if first else None)

        plt.plot(xs, ys, 'b-', 0.2)

        plt.title(f"GCAS network prediction {statenames[output_var]}")
        plt.savefig(f'gcas_predict{output_var}_{statenames[output_var]}.png')

        plt.clf()

    #compare(sess)

def compare(sess):
    """compare with data file"""
    
    data = np.load('test_data.npz')

    x = data['x']
    y = data['y']
    y_est = data['y_est']

    for i in range(50):
        x0 = x[i]
        y0 = y[0][i]
        y_est0 = y_est[i]

        mine = run_network(sess, x0)

        #print(mine)
        #print(y_est0)
        print(f"percent error: {100 * np.linalg.norm(mine - y_est0) / np.linalg.norm(y_est0)}")

if __name__ == "__main__":
    main()
