'''
generate spec files for prob density reachability

Stanley Bak, June 2022
'''


import sys
import random

from math import pi

import numpy as np

def write_spec_file(filename, note, init_box, mat, rhs):
    """write a spec file"""

    num_io = init_box.shape[0]

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'; {note}\n\n')

        for i in range(num_io):
            f.write(f'(declare-const X_{i} Real)\n')

        f.write('\n')

        for i in range(num_io):
            f.write(f'(declare-const Y_{i} Real)\n')

        f.write('\n; Input constraints:\n')

        for index, (lb, ub) in enumerate(init_box):
            f.write(f'(assert (<= X_{index} {ub}))\n')
            f.write(f'(assert (>= X_{index} {lb}))\n\n')

        # targetted misclasification
        f.write('\n; Output constraints (encoding the conditions for a property counter-example):\n')

        for index, row in enumerate(mat):
            var = np.argmax(np.abs(row))
            val = row[var]
            assert val in [-1, 1], f"row can only have 1 or -1 or 0 entries: {row}"
            row[var] = 0
            assert np.max(np.abs(row)) == 0, f"mulitple nonzeros? in row: {row}"

            if val == 1:
                f.write(f'(assert (<= Y_{var} {rhs[index]}))\n')
            else:
                f.write(f'(assert (>= Y_{var} {-rhs[index]}))\n')

        f.write('\n')

def make_vdp_spec(index):
    """main vanderpol spec file"""

    rad = 0.2 + 0.6 * random.random()
    log_prob = 0.15 + 0.15 * random.random()

    trange = [0.0, 5.0]
    init_box = np.array([[-2.5, 2.5], [-2.5, 2.5], trange], dtype=float)

    mat = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0]], dtype=float)
    rhs = np.array([rad, rad, rad, rad, -log_prob], dtype=float)

    note = f'VDP spec with rad {rad} and log_prob {log_prob}'
    filename = f'vdp_{index}.vnnlib'

    write_spec_file(filename, note, init_box, mat, rhs)

    return filename

def make_robot_spec(index):
    """main robot spec file"""

    rad = 0.3 * random.random()
    log_prob = 0.05 + 0.25 * random.random()

    mat = np.array([[0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [-1, 0, 0, 0, 0]], dtype=float)
    rhs = np.array([rad, rad, rad, rad, -log_prob], dtype=float)

    tmax = 0.05 * 50
    init_box = np.array([(-1.8, -1.2), (-1.8, -1.2), (0, pi/2), (1.0, 1.5), (0, tmax)], dtype=float)

    note = f'Robot spec with rad {rad} and log_prob {log_prob}'
    filename = f'robot_{index}.vnnlib'

    write_spec_file(filename, note, init_box, mat, rhs)

    return filename

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

def make_gcas_spec(index):
    """main gcas spec file"""

    theta_max = -1 + 0.5 * random.random()
    scale = 1.0 + random.random() * 3

    note = f"gcas with theta_max {theta_max} and scale {scale}"

    theta_range = (theta_max, 0)

    scaling_dict = get_scaling()

    trange = (0, 105 * 1/30)
    # yue paper init set
    tuples_list = [(560.0040, 599.9999), # vt
        (0.0750, 0.1000), # alpha
        (scale * -0.0100, scale *0.0100), # beta
        (-0.1000, -0.0750), # phi (roll angle)

        #(-0.9996, -0.5000), # theta!
        #(-0.88, 0), # theta!
        theta_range,

        (scale * -0.0100, scale * 0.0100),
        (scale * -0.0100, scale * 0.0100),
        (scale * -0.0100, scale * 0.0100),
        (scale * -0.0100, scale * 0.0100),
        (scale * -0.0100, scale * 0.0100),
        (scale * -0.0100, scale * 0.0100),

        #(1150.0081, 1199.9975), # alt
        (1100, 1200), # alt

        (0.0002, 0.9998), # pow
        trange]

    # scale input
    for i, tup in enumerate(tuples_list):
        lb = (tup[0] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]
        ub = (tup[1] - scaling_dict['in_means'][i]) / scaling_dict['in_stds'][i]

        tuples_list[i] = (lb, ub)
    
    init_box = np.array(tuples_list, dtype=float)

    #####################
    log_prob = 0
    min_alt = 0 # 42.9

    alt_index = 12
    prob_index = 0
    
    row1 = [1 if i == alt_index else 0 for i in range(14)]
    row2 = [-1 if i == prob_index else 0 for i in range(14)]
    
    mat = np.array([row1, row2], dtype=float)
    rhs = np.array([min_alt, -log_prob], dtype=float)

    # apply output scaling to rhs
    rhs_indices = [alt_index, prob_index]
    
    for rhs_index, (i, x) in enumerate(zip(rhs_indices, rhs)):
        newx = (x - scaling_dict['out_means'][i]) / scaling_dict['out_stds'][i]
        rhs[rhs_index] = newx

    filename = f'gcas_{index}.vnnlib'

    write_spec_file(filename, note, init_box, mat, rhs)

    return filename

def main():
    """main entry point"""

    assert len(sys.argv) == 2, "expected 1 arg: <seed>"
    random.seed(int(sys.argv[1]))

    with open('instances.csv', 'w', encoding='utf-8') as f:

        for i in range(12):
            spec_path = make_vdp_spec(i)
            f.write(f'vdp.onnx,{spec_path},600\n')

        for i in range(12):
            spec_path = make_robot_spec(i)
            f.write(f'robot.onnx,{spec_path},600\n')

        for i in range(12):
            spec_path = make_gcas_spec(i)
            f.write(f'gcas.onnx,{spec_path},600\n')

if __name__ == '__main__':
    main() 
