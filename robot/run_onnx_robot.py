"""
Stanley Bak

Test Vanderpol onnx output
"""

from collections import namedtuple
from math import pi

import onnxruntime as ort
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.relu = nn.ReLU()
        self.args = args

        self.linear_list = nn.ModuleList()
        input_dim = 4  # x,y,th,v
        output_dim = 2  # omega, accel
        self.linear_list.append(nn.Linear(input_dim, args.hiddens[0]))
        for i,hidden in enumerate(args.hiddens):
            if i==len(args.hiddens)-1:  # last layer
                self.linear_list.append(nn.Linear(args.hiddens[i], output_dim))
            else:  # middle layers
                self.linear_list.append(nn.Linear(args.hiddens[i], args.hiddens[i + 1]))

    def forward(self, x):
        for i, hidden in enumerate(self.args.hiddens):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[len(self.args.hiddens)](x)
        if self.args.output_type=="tanh":
            x = nn.Tanh()(x) * self.args.tanh_gain
        return x

class Benchmark:
    def __init__(self, mock_args, args):
        mock_args.hiddens = [32, 32]
        mock_args.output_type = "tanh"
        mock_args.tanh_gain = 4.0
        #actor = Actor(mock_args)
        #actor.load_state_dict(torch.load(args.pretrained_path))
        #actor.eval()

        self.n_dim = 4  # TODO(x, y, th, v)
#        self.actor = actor
        self.args = args

#    def get_u_and_du(self, state):
#        state_tensor = torch.from_numpy(state).float()
#        state_tensor.requires_grad = True
#        u_tensor = self.actor(state_tensor)
#        du1 = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=state_tensor,
#                                  grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
#        du2 = torch.autograd.grad(outputs=u_tensor[:, 1:2], inputs=state_tensor,
#                                  grad_outputs=torch.ones_like(u_tensor[:, 1:2]), retain_graph=True)[0]

#        du1dth = du1.detach().numpy()[:, 2:3]
#        du2dv = du2.detach().numpy()[:, 3:4]
#        uv = u_tensor.detach().numpy()
#        return uv, [du1dth, du2dv]

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, state, uv):
        dx = state[:, 3] * np.cos(state[:, 2])
        dy = state[:, 3] * np.sin(state[:, 2])
        dth = uv[:, 0]
        dv = uv[:, 1]
        new_state = np.array(state)
        new_state[:, 0] += dx * self.args.dt
        new_state[:, 1] += dy * self.args.dt
        new_state[:, 2] += dth * self.args.dt
        new_state[:, 3] += dv * self.args.dt

        return new_state

    # TODO (needed)
    def get_u_du_new_s(self, state):
        for i in range(self.args.sim_steps):
            uv, du_sum = self.get_u_and_du(state)
            new_state = self.get_next_state(state, uv)
            state = new_state
        return uv, du_sum, state

    # TODO (needed)
    def get_nabla(self, state, u, du_cache):
        du1dth, du2dv = du_cache
        return (du1dth + du2dv).flatten()

    # TODO(dbg for ode)
    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]
        state_tensor = torch.from_numpy(state.reshape((1, ndim))).float()
        state_tensor.requires_grad = True
        u_tensor = self.actor(state_tensor)
        du1 = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        du2 = torch.autograd.grad(outputs=u_tensor[:, 1:2], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 1:2]), retain_graph=True)[0]
        du1dth = du1.detach().numpy()[0, 2:3]
        du2dv = du2.detach().numpy()[0, 3:4]
        nabla = (du1dth + du2dv)
        drho = - nabla * rho

        # dx = state[:, 3] * np.cos(state[:, 2])
        # dy = state[:, 3] * np.sin(state[:, 2])
        # dth = uv[:, 0]
        # dv = uv[:, 1]
        #
        u = u_tensor.detach().cpu().numpy().flatten()
        dx = np.zeros((ndim,))
        dx[0] = state[3] * np.cos(state[2])
        dx[1] = state[3] * np.sin(state[2])
        dx[2] = u[0]
        dx[3] = u[1]

        dxdrho = np.zeros(ndim+1)
        dxdrho[:ndim] = dx
        dxdrho[-1] = drho
        return dxdrho  #np.concatenate((dx, drho))

class MockArgs:
    pass


def run_network(sess, x, stdout=False):
    'run the network and return the output'

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, len(x))

    
    outputs = sess.run(None, {'input': in_array})

    if stdout:
        print(f"stdout: {in_array} -> {outputs[0][0]}")

    return outputs[0][0]

def main():
    """main entry point"""

    sess = ort.InferenceSession("model499000.onnx")
    #sess = ort.InferenceSession("model0.onnx")

    first = True

    ranges = [(-1.8, -1.2), (-1.8, -1.2), (0, pi/2), (1.0, 1.5)]
    #ranges = [(-1.8, 0), (-1.8, -1.2), (0, pi/2), (1.0, 1.5)]
    parts = 10

    first = True

    for x in np.linspace(*ranges[0], parts):
        for y in np.linspace(*ranges[1], parts):
            for theta in np.linspace(*ranges[2], parts):
                for v in np.linspace(*ranges[3], parts):

                    if first:
                        # concrete counterexamples
                        #x, y, theta, v = -1.8, -1.4897948306982278, 0.8029154399554316, 1.5 #, 0.982018266144939
                        x, y, theta, v, terror = -1.5391385350032363, -1.340879522599621, 0.7820428854489013, 1.3262732018272776, 0.8165131580548769

                    pt = [x, y, theta, v]

                    state = np.array([pt], dtype=float)

                    dt = 0.05
                    steps = 50
                    xs = []
                    ys = []
                    Args = namedtuple('Args', ['dt'])
                    args = Args(dt)

                    mock_args = MockArgs()
                    sys = Benchmark(mock_args, args)
                    sys_xs = []
                    sys_ys = []

                    for t in np.arange(0, dt * steps, dt):
                        res = run_network(sess, [*pt, t], stdout=False)

                        xs.append(res[1])
                        ys.append(res[2])

                        #sys_xs.append(state[0, 0])
                        #sys_ys.append(state[0, 1])

                        #print(f"state ({state.shape}): {state}")

                        #prev_state = state.copy()
                        #state = sys.get_next_state(state, 0)
                        #print(f"from {prev_state} to {state}")

            #plt.plot(sys_xs, sys_ys, 'r-', lw=0.3 if first else 0.1, label='actual' if first else None)

            color = 'r-' if first else 'b-'
            plt.plot(xs, ys, color, lw=0.3 if first else 0.1, label='predicted' if first else None)

            if first:
                t = terror

                res = run_network(sess, [*pt, t], stdout=False)
                print(f"first res: {res}")

                plt.plot(pt[0], pt[1], 'ro')
                plt.plot(res[1], res[2], 'ro')

            first = False

    x0, x1 = ranges[0]
    y0, y1 = ranges[1]
    
    xs = [x0, x0, x1, x1, x0]
    ys = [y0, y1, y1, y0, y0]
    plt.plot(xs, ys, 'k-', lw=2)

    plt.title("Robot sim vs network prediction")
    plt.legend()
    plt.savefig('robot_test.png')

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
