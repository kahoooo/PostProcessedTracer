import argparse
from collections import defaultdict

import numpy as np

from frame import Frame
from utils import serialize


def main():
    parser = argparse.ArgumentParser(description='Combine output from particles per frame to history of particles. '
                                                 'Also extract density, velocity and temperature information.')
    parser.add_argument('frames', action='store', nargs='+', type=str,
                        help='primitives variables in athdf format')
    parser.add_argument('-f', '--format', action='store', type=str, default='particle.%05d.npy',
                        help='output filename format containing one format specifier')
    args = parser.parse_args()

    # construct a sorted list of frames
    frames = [Frame(filename) for filename in args.frames]
    frames.sort()

    mesh2mb_cache = dict()
    interpcc_cache = dict()
    record = defaultdict(list)

    for frame in frames:
        print('Working on frame', frame.filename)
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

        frame.load(['rho', 'vel1', 'vel2', 'vel3', 'press'])

        quantities = np.stack([frame.data['vel1'], frame.data['vel2'], frame.data['vel3'],
                               frame.data['rho'], frame.data['press'] / frame.data['rho']])

        for pid, (x1, x2, x3) in zip(pids, meshs):
            mb = mesh2mb(x1, x2, x3)
            q = interpcc(quantities, mb, x1, x2, x3)
            record[pid].append(np.concatenate([[time, x1, x2, x3], q]))

        frame.unload()

    for pid in record:
        np.save(args.format % pid, record[pid])


if __name__ == '__main__':
    main()
