import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from postprocessedtracer.frame import Frame
from postprocessedtracer.utils import serialize


def main():
    parser = argparse.ArgumentParser(description='Combine output from particles per frame to history of particles. '
                                                 'Also extract density, velocity and temperature information.')
    parser.add_argument('--frames', action='store', nargs='+', type=str,
                        help='primitives variables in athdf format')
    parser.add_argument('-y', '--npy', action='store', type=str,
                        help='output in npy format and set the filename format (with one format specifier)')
    parser.add_argument('-z', '--npz', action='store', type=str, default='particles.npz',
                        help='output in npz format and set the filename')
    parser.add_argument('--ix1', action='store', type=str, default='outflow',
                        help='x1 inner boundary')
    parser.add_argument('--ox1', action='store', type=str, default='outflow',
                        help='x1 outer boundary')
    parser.add_argument('--ix2', action='store', type=str, default='polar',
                        help='x2 inner boundary')
    parser.add_argument('--ox2', action='store', type=str, default='reflecting',
                        help='x2 outer boundary')
    parser.add_argument('--ix3', action='store', type=str, default='periodic',
                        help='x3 inner boundary')
    parser.add_argument('--ox3', action='store', type=str, default='periodic',
                        help='x3 outer boundary')
    parser.add_argument('--ncols', action='store', type=int,
                        help='number of columns used to print progress bar')
    args = parser.parse_args()

    # construct a sorted list of frames
    with tqdm(args.frames, ncols=args.ncols,
              bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}') as t:
        frames = []
        for filename in t:
            t.set_description_str(f'Reading header from {filename}')
            frames.append(Frame(filename, boundaries=((args.ix1, args.ox1),
                                                      (args.ix2, args.ox2),
                                                      (args.ix3, args.ox3)), num_ghost=2))
    frames.sort()

    mesh2mb_cache = dict()
    interpcc_cache = dict()
    record = defaultdict(list)

    with tqdm(frames, ncols=args.ncols,
              bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}') as t:
        for frame in t:
            t.set_description_str(f'Working on {filename}')
            h = dict(frame.header)
            for key in ['Time', 'NumCycles']:
                del h[key]

            hash_key = serialize(h)
            if hash_key not in mesh2mb_cache:
                mesh2mb_cache[hash_key] = frame.mesh_position_to_meshblock_id
            if hash_key not in interpcc_cache:
                interpcc_cache[hash_key] = frame.interpolate_cell_centered

            mesh2mb = mesh2mb_cache[hash_key]
            interpcc = interpcc_cache[hash_key]

            par = np.load(frame.filename + '.npz')
            time = par['time'][()]
            pids = par['pids']
            meshs = par['meshs']
            flags = par['flags']

            frame.load(['rho', 'vel1', 'vel2', 'vel3', 'press', 'int_rho_dr'])
            # frame.patch_boundary(['press'])

            quantities = np.stack([frame.data['vel1'], frame.data['vel2'], frame.data['vel3'],
                                   frame.data['rho'], frame.data['press'] / frame.data['rho'],
                                   frame.data['int_rho_dr']])

            for pid, (x1, x2, x3), flag in zip(pids, meshs, flags):
                mb = mesh2mb(x1, x2, x3)
                q = interpcc(quantities, mb, x1, x2, x3)
                q[-1] = np.clip(q[-1], 0.0, None)
                record[flag.decode()+str(pid)].append(np.concatenate([[time, x1, x2, x3], q]))

            frame.unload()

    if args.npy is not None:
        for pid in record:
            np.save(args.npy % pid, record[pid])
    else:
        np.savez(args.npz, **{str(k): np.array(v) for k, v in record.items()})


if __name__ == '__main__':
    main()
