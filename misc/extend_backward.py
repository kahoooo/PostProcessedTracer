import argparse

import numpy as np
from tqdm import tqdm

from postprocessedtracer.extend_history.extend import Extend
from postprocessedtracer.extend_history.utils import *


def main():
    parser = argparse.ArgumentParser(description='Convert between two distribution')
    parser.add_argument('--in', '-i', action='store', required=True, type=str,
                        help='npz file input')
    parser.add_argument('--out', '-o', action='store', required=True, type=str,
                        help='npz file output')
    parser.add_argument('--time', '-t', action='store', required=True, type=float,
                        help='time between initial and final distribution')
    parser.add_argument('--steps', '-s', action='store', required=True, type=int,
                        help='number of steps between initial and final distribution')
    parser.add_argument('--ncols', action='store', type=int,
                        help='number of columns used to print progress bar')
    args = vars(parser.parse_args())

    argtime = args['time']
    argsteps = args['steps']

    extend = Extend(
        singular_isothermal_sphere(m_0=130.0 * 5 / 8, r_0=2.0503493309826641),
        uniform_sphere(rho_0=7.855735488699885e-03),  # 7.4603879725194059e-03
        tanh_velocity(0.01),
        dt=argtime, mmax=130.0 * 5 / 8, omega_i=31.553241 * 0.5, steps=argsteps)

    par_in = np.load(args['in'])
    par_out = dict()
    with tqdm(par_in.items(), ncols=args['ncols']) as t:
        for key, value in t:
            # columns: time, r, theta, phi, vr, vtheta, vphi, rho, T
            time, r, phi, vr, vphi, rho = extend.history(r_i=value[0, 1])

            nrow = par_in[key].shape[0] + argsteps
            ncol = par_in[key].shape[1]
            value_out = np.empty((nrow, ncol), dtype=float)
            value_out[:argsteps, 0] = -time[:0:-1]
            value_out[:argsteps, 1] = r[:0:-1]
            value_out[:argsteps, 2] = value[0, 2]
            value_out[:argsteps, 3] = value[0, 3] - phi[:0:-1]
            value_out[:argsteps, 4] = -vr[:0:-1]
            value_out[:argsteps, 5] = 0.0
            value_out[:argsteps, 6] = vphi[:0:-1] * np.sin(value[0, 2])
            value_out[:argsteps, 7] = rho[:0:-1]
            value_out[:argsteps, 8] = value[0, 8]

            value_out[argsteps:, :] = value
            value_out[:, 0] -= value_out[0, 0]
            value_out[:, 3] -= value_out[0, 3]
            value_out[:, 3] %= 2 * np.pi

            par_out[key] = value_out

    # get column density at all times
    keys = list(par_out.keys())
    keys.sort(key=lambda k: par_out[k][argsteps, 1])
    r_i = [par_out[k][argsteps, 1] for k in keys]
    m = list(np.vectorize(extend.lamb_m_i)(r_i))
    m.append(extend.mmax)
    with tqdm(list(enumerate(time[1:])), ncols=args['ncols']) as t:
        for i, timenow in t:
            dsigma = np.vectorize(extend.lamb_sigma)(timenow, m[:-1], m[1:])
            sigma = np.cumsum(dsigma[::-1])[::-1]
            for k, s in zip(keys, sigma):
                par_out[k][argsteps-i-1, 9] = s

    np.savez(args['out'], **par_out)


if __name__ == '__main__':
    main()
