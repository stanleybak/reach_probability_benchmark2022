'''
Stanley Bak

run the gcas system, producing a 3d animation

pass a command line argument (ending with .mp4) to save a video instead of plotting to the screen
'''

import math
import sys

from numpy import deg2rad

import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import anim3d, plot

from gcas_autopilot import GcasAutopilot

def simulate():
    'sim system and return res'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 6200        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.7         # Pitch angle from nose level (rad)
    psi = 0.8 * math.pi   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 15.0 #15 # simulation time

    print("!! init override !!")
    init = [560.004, 0.075, -0.008766038321942879, -0.1, -0.9996, -0.01, -0.01, -0.01, -0.008571254908063456, -0.01, -0.01, 1150.0081, 0.00019999999999997797]
    #tmax = 3.51 # simulation time

    ap = GcasAutopilot(init_mode='waiting', stdout=True)

    ap.waiting_time = 5
    ap.waiting_cmd[1] = 2.2 # ps command

    # custom gains
    ap.cfg_k_prop = 1.4
    ap.cfg_k_der = 0
    ap.cfg_eps_p = deg2rad(20)
    ap.cfg_eps_phi = deg2rad(15)

    step = 1/30
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True, integrator_str='rk45')

    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds")

    return res

def main():
    'main function'
    
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res = simulate()

    plot.plot_attitude(res, figsize=(12, 10))
    plt.savefig('gcas_attitude.png')
    plt.close()
    
    anim3d.make_anim(res, filename, elev=15, azim=-150)

if __name__ == '__main__':
    main()
