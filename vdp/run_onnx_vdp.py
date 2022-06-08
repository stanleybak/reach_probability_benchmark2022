"""
Stanley Bak

Test Vanderpol onnx output
"""

from collections import namedtuple

import onnxruntime as ort
import numpy as np

import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, mock_args, args):
        self.n_dim = 2  # TODO(x, y, k1, k2)
        self.args = args
        self.mu=1.0

    def get_u_and_du(self, x):
        return None, None

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = self.mu * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
        
        print(f"dxdt: {dxdt}")
        
        return x + dxdt * self.args.dt

    # TODO (needed)
    def get_u_du_new_s(self, x):
        for _ in range(self.args.sim_steps):
            u, du_cache = self.get_u_and_du(x)
            new_x = self.get_next_state(x, u)
            x = new_x
        return u, du_cache, x

    # TODO (needed)
    def get_nabla(self, x, u, du_cache):
        return self.mu * (1 - x[:, 0] ** 2)

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        x = state.reshape((1, ndim))
        rho = x_rho[-1]
        nabla = self.get_nabla(x, None, None)
        drho = - nabla * rho

        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = self.mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
        
        

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dxdt[0,:]
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))

def run_network(sess, x, stdout=False):
    'run the network and return the output'

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 3)
    outputs = sess.run(None, {'input': in_array})

    if stdout:
        print(f"stdout: {in_array} -> {outputs[0][0]}")

    return outputs[0][0]

def main():
    """main entry point"""

    sess = ort.InferenceSession("vdp.onnx")
    #sess = ort.InferenceSession("model0.onnx")

    first = True
    start = [1, 1] # start point 
    delta = 0.5 # start area
    single = False

    for startx in np.linspace(start[0] - delta, start[0] + delta, 10):
        for starty in np.linspace(start[1] - delta, start[1] + delta, 10):
            
            pt = [startx, starty]

            if first:
                # hardcoded state
                pt = [0.17367119599879288, 0.19598037238019206]

            state = np.array([pt], dtype=float)

            dt = 0.1
            steps = 50
            xs = []
            ys = []
            Args = namedtuple('Args', ['dt'])
            args = Args(dt)

            sys = Benchmark(None, args)
            sys_xs = []
            sys_ys = []

            for t in np.arange(0, dt * steps, dt):
                res = run_network(sess, [pt[0], pt[1], t], stdout=False)

                if t >= 0:#4.0:
                    xs.append(res[1])
                    ys.append(res[2])

                    sys_xs.append(state[0, 0])
                    sys_ys.append(state[0, 1])

                prev_state = state.copy()
                state = sys.get_next_state(state, 0)
                print(f"from {prev_state} to {state}")

            if single:
                plt.plot([pt[0]], [pt[1]], 'ko', label='start' if first else None)
                plt.plot([xs[0]], [ys[0]], 'bo', label='predicted start' if first else None)

                plt.plot(xs, ys, 'b-', label='predicted' if first else None)
                plt.plot(sys_xs, sys_ys, 'k-', label='actual' if first else None)
                break

            plt.plot(sys_xs, sys_ys, 'r-', lw=0.3 if first else 0.1, label='actual' if first else None)
            plt.plot(xs, ys, 'b-', lw=0.3 if first else 0.1, label='predicted' if first else None)

            first = False

        if single:
            break

    if not single:
        xs = [start[0] - delta, start[0] + delta, start[0] + delta, start[0] - delta, start[0] - delta]
        ys = [start[1] - delta, start[1] - delta, start[1] + delta, start[1] + delta, start[1] - delta]
        plt.plot(xs, ys, 'k-', lw=2)

    plt.title("Vanderpol sim vs network prediction")
    plt.legend()
    plt.savefig('vdp_test.png')

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
