'''
Stanley Bak

should match 'GCAS' scenario from matlab version
'''

import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from gcas_autopilot import GcasAutopilot

def main():
    'main function'

    ### Initial Conditions ###
    power = 0.5045 # 8 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1200        # altitude (ft)
    vt = 560          # initial velocity (ft/sec)
    phi = -0.0877 #-math.pi/8           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    print("!! init override !!")
    init = [560.004, 0.075, -0.008766038321942879, -0.1, -0.5996, -0.01, -0.01, -0.01, -0.008571254908063456, -0.01, -0.01, 1150.0081, 0.00019999999999997797]
    
    tmax = 3.51 # simulation time

    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

    step = 1/30
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)

    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")
    print(f"length: {res['states'].shape}")

    statenames = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']

    for i, name in enumerate(statenames):
        plot.plot_single(res, name, title=f'Var {i} - {name}')
        filename = f'real{i}_{name}.png'
        plt.savefig(filename)
        plt.clf()
        
        print(f"Made {filename}")

    plt.clf()
    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
