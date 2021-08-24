import argparse
import itertools as it
import numba as nb
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt

from frame import Frame
from particles import Particles
from integrator import VanLeer2
from utils import poisson_disk_sampler

import warnings
warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


def plot_particles(par: Particles):
    x = par.meshs[:, 0] * np.sin(par.meshs[:, 1])
    y = par.meshs[:, 0] * np.cos(par.meshs[:, 1])
    plt.scatter(x, y, s=1)
    plt.scatter(x, -y, s=1)


def plot_background(frame: Frame):
    xf = frame.header['x1f'][:, :, None] * np.sin(frame.header['x2f'][:, None, :])
    yf = frame.header['x1f'][:, :, None] * np.cos(frame.header['x2f'][:, None, :])
    xv = frame.header['x1v'][:, :, None] * np.sin(frame.header['x2v'][:, None, :])
    yv = frame.header['x1v'][:, :, None] * np.cos(frame.header['x2v'][:, None, :])

    rho_vmin = frame.data['rho'].min()
    rho_vmax = frame.data['rho'].max()
    vel1_vmin = frame.data['vel1'].min()
    vel1_vmax = frame.data['vel1'].max()
    vel1_vmax = max(vel1_vmax, -vel1_vmin)
    vel1_vmin = -vel1_vmax
    for mb in range(frame.header['NumMeshBlocks']):
        plt.pcolormesh(xf[mb, 2:-2, 2:-2], yf[mb, 2:-2, 2:-2], frame.data['rho'][mb, 0, 2:-2, 2:-2].T,
                       norm=colors.LogNorm(vmin=rho_vmin, vmax=rho_vmax), zorder=0, cmap='jet')
        plt.pcolormesh(xf[mb, 2:-2, 2:-2], -yf[mb, 2:-2, 2:-2], frame.data['vel1'][mb, 0, 2:-2, 2:-2].T,
                       norm=colors.Normalize(vmin=vel1_vmin, vmax=vel1_vmax), zorder=0, cmap='bwr')
    plt.gca().set_aspect(1)


def main():
    parser = argparse.ArgumentParser(description='Post-processed tracer particles')
    parser.add_argument('frames', action='store', nargs='+', type=str,
                        help='primitives variables in athdf format')
    parser.add_argument('--keyframes', '-k', action='store', nargs='*', type=str,
                        help='key frames when new particles is inserted')
    parser.add_argument('--backward', '-b', action='store_true',
                        help='integrate backward in time')
    parser.add_argument('--seed', '-s', action='store', type=int,
                        help='seed for random number generation')
    args = parser.parse_args()

    # construct a sorted list of frames in the order of integration
    frames = [Frame(filename, boundaries=(('none', 'none'),
                                          ('polar', 'reflecting'),
                                          ('periodic', 'periodic'))) for filename in args.frames]
    frames.sort(reverse=args.backward)

    # seed both numpy and numba with the same seed
    if args.seed is not None:
        @nb.njit
        def seed_numba(seed):
            np.random.seed(seed)
        seed_numba(args.seed)
        np.random.seed(args.seed)

    # particle positions in mesh coordinates, no particles initially
    par = Particles(0)
    integrator = VanLeer2(cfl=0.1, cfl_inactive=0.01)

    for first, second in it.zip_longest(frames, frames[1:]):
        if first.filename in args.keyframes:
            print('Generating particles for', first.filename)
            poisson_disk_sampler(first, par, radius=0.8)
            print(f'{par.size} particles')

        np.savez(first.filename + '.npz',
                 frame=first.filename, time=first.time,
                 pids=par.pids, meshs=par.meshs)

        if second is not None:
            print('Reading data...')
            first.load(['vel1', 'vel2', 'vel3'])
            second.load(['vel1', 'vel2', 'vel3'])
            integrator.integrate(first, second, par)
            first.unload()

    frame = frames[-1]
    frame.load(['rho', 'vel1', 'vel2', 'vel3'])
    plot_background(frame)
    plot_particles(par)
    plt.show()


if __name__ == '__main__':
    main()
