import argparse
import itertools as it
import warnings

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from matplotlib import colors
from tqdm import tqdm

from frame import Frame
from integrator import VanLeer2
from particles import Particles
from utils import poisson_disk_sampler

warnings.filterwarnings("ignore", category=nb.NumbaExperimentalFeatureWarning)


def plot_particles(par: Particles, zorder=2):
    x = par.meshs[:, 0] * np.sin(par.meshs[:, 1])
    y = par.meshs[:, 0] * np.cos(par.meshs[:, 1])
    plt.scatter(x, y, s=1, zorder=zorder)
    plt.scatter(x, -y, s=1, zorder=zorder)


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
    parser.add_argument('--frames', action='store', nargs='+', type=str,
                        help='primitives variables in athdf format')
    parser.add_argument('--keyframes', '-k', action='store', nargs='*', type=str,
                        help='key frames when new particles is inserted')
    parser.add_argument('--backward', '-b', action='store_true',
                        help='integrate backward in time')
    parser.add_argument('--sample_mass', action='store', type=float, default=0,
                        help='minimum particle distance in mass-space')
    parser.add_argument('--sample_space', action='store', type=float, default=0,
                        help='minimum particle distance')
    parser.add_argument('--seed', '-s', action='store', type=int,
                        help='seed for random number generation')
    parser.add_argument('--ncols', action='store', type=int,
                        help='number of columns used to print progress bar')
    args = parser.parse_args()

    # construct a sorted list of frames in the order of integration
    with tqdm(args.frames, ncols=args.ncols,
              bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}') as t:
        frames = []
        for filename in t:
            t.set_description_str(f'Reading header from {filename}')
            frames.append(Frame(filename, boundaries=(('none', 'none'),
                                                      ('polar', 'reflecting'),
                                                      ('periodic', 'periodic'))))
        frames.sort(reverse=args.backward)

    # seed both numpy and numba with the same seed
    if args.seed is not None:
        @nb.njit
        def seed_numba(seed):
            np.random.seed(seed)

        seed_numba(args.seed)
        np.random.seed(args.seed)

    # particle positions in mesh coordinates, no particles initially, then sample weighted by mass
    par = Particles(0)
    if args.sample_mass > 0:
        ngh = frames[0].num_ghost
        ndim = frames[0].num_dimension
        slc = (slice(None),) * (4 - ndim) + (slice(ngh, -ngh),) * ndim
        frames[0].load(['rho'])
        rho = frames[0].data['rho']
        dvol = frames[0].get_finite_volume()
        mass_per_cell = (rho * dvol)[slc]
        mindist = (args.sample_mass / mass_per_cell) ** (1 / ndim)
        nsample = np.log(mass_per_cell.max() / mass_per_cell.sum()) / np.log(0.99)
        poisson_disk_sampler(frames[0], par, mindist=mindist, seed=nsample)

    integrator = VanLeer2(cfl=0.1, cfl_inactive=0.01)

    observations = np.zeros(len(frames) + 1, dtype=int)
    with tqdm(smoothing=0.1, ncols=args.ncols,
              bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{eta}] {desc}') as pbar:
        for i, (first, second) in enumerate(it.zip_longest(frames, frames[1:])):
            if args.sample_space > 0 and first.filename in args.keyframes:
                pbar.set_description_str(f'Generating particles for {first.filename}')
                poisson_disk_sampler(first, par, radius=args.sample_space)

            np.savez(first.filename + '.npz',
                     frame=first.filename, time=first.time,
                     pids=par.pids, meshs=par.meshs)

            if second is not None:
                t.set_description_str(f'Reading data from {first.filename}')
                first.load(['vel1', 'vel2', 'vel3'])
                second.load(['vel1', 'vel2', 'vel3'])
                integrator.integrate(first, second, par, pbar=pbar)
                first.unload()

            observations[i + 1] = observations[i] + par.size
            if i != len(frames) - 1:
                x = np.arange(1, i + 2)
                y = observations[1:i + 2]
                fitted = np.poly1d(np.polyfit(x, y, min(3, i)))
                factor = np.max(y / fitted(x))
                pbar.total = int(fitted(len(frames)) * factor)
            else:
                pbar.total = observations[-1]
            pbar.update(par.size)

    frame = frames[-1]
    frame.load(['rho', 'vel1', 'vel2', 'vel3'])
    plot_background(frame)
    plot_particles(par)
    plt.show()


if __name__ == '__main__':
    main()
