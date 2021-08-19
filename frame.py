import numba as nb
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


def _convert_type(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, bytes):
        return x.decode('ascii', 'replace')
    if isinstance(x, np.ndarray):
        if issubclass(np.obj2sctype(x.dtype), bytes):
            return _convert_type(x.tolist())
        return x.astype(np.obj2sctype(x.dtype))
    if isinstance(x, list):
        return list(map(_convert_type, x))
    return x


@dataclass(init=False, order=True)
class Frame:
    filename: str = field(compare=False)
    time: float = field(repr=False)
    header: dict = field(compare=False, repr=False)
    data: dict = field(compare=False, repr=False)
    num_ghost: int = field(compare=False, repr=False)
    num_dimension: int = field(compare=False, repr=False)
    boundaries: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = field(compare=False, repr=False)

    mesh_position_to_fractional_position_root: Callable = field(compare=False, repr=False)
    mesh_position_to_fractional_position_meshblock: Callable = field(compare=False, repr=False)
    mesh_position_to_meshblock_id: Callable = field(compare=False, repr=False)
    mesh_position_to_global_indices: Callable = field(compare=False, repr=False)
    mesh_position_to_local_indices: Callable = field(compare=False, repr=False)
    global_indices_to_mesh_position: Callable = field(compare=False, repr=False)
    velocity_to_derivatives: Callable = field(compare=False, repr=False)
    interpolate_cell_centered: Callable = field(compare=False, repr=False)
    apply_boundaries: Callable = field(compare=False, repr=False)

    def __init__(self, filename, boundaries=None):
        self.filename = filename

        if boundaries is None:
            self.boundaries = (('none',) * 2, ) * 3
        else:
            try:
                (ix1, ox1), (ix2, ox2), (ix3, ox3) = boundaries
                self.boundaries = ((str(ix1), str(ox1)), (str(ix2), str(ox2)), (str(ix3), str(ox3)))
            except ValueError:
                raise ValueError('boundaries has to be in the form: ((ix1, ox1), (ix2, ox2), (ix3, ox3))')

        self._load_header()
        self.time = self.header['Time']
        self.data = {}

        self._prepare_functions()

    def load(self, quantities=None):
        import h5py

        if quantities is None:
            quantities = list(self.header['VariableNames'])

        num_variables = self.header['NumVariables']
        dataset_prefix = np.cumsum(num_variables)
        dataset_offsets = dataset_prefix - dataset_prefix[0]

        with h5py.File(self.filename, 'r') as f:
            dataset_names = self.header['DatasetNames']
            variable_names = self.header['VariableNames']

            for q in quantities:
                if q in self.data:
                    continue

                if q not in variable_names:
                    raise RuntimeError(f'Quantity "{q}" not found, '
                                       f'available quantities include {variable_names}')

                variable_index = variable_names.index(q)
                dataset_index = np.searchsorted(dataset_prefix, variable_index)
                variable_index -= dataset_offsets[dataset_index]
                self.data[q] = _convert_type(f[dataset_names[dataset_index]][variable_index])

    def unload(self):
        self.data = {}

    def _load_header(self):
        import h5py

        with h5py.File(self.filename, 'r') as f:
            self.header = {}

            for key in f.attrs:
                self.header[key] = _convert_type(f.attrs[key])

            dataset_names = self.header['DatasetNames']
            for key in f.keys():
                if key in dataset_names:
                    continue
                self.header[key] = _convert_type(f[key][:])

    def _prepare_functions(self):
        nx_root = self.header['RootGridSize']
        nx_meshblock = self.header['MeshBlockSize']
        maxlevel = self.header['MaxLevel']
        llocs = self.header['LogicalLocations']
        levels = self.header['Levels']

        x1f = self.header['x1f']
        x2f = self.header['x2f']
        x3f = self.header['x3f']
        x1v = self.header['x1v']
        x2v = self.header['x2v']
        x3v = self.header['x3v']

        nx1rt, nx2rt, nx3rt = nx_root
        x1minrt, x1maxrt, x1ratrt = self.header['RootGridX1']
        x2minrt, x2maxrt, x2ratrt = self.header['RootGridX2']
        x3minrt, x3maxrt, x3ratrt = self.header['RootGridX3']
        x1ratnxrt, x2ratnxrt, x3ratnxrt = x1ratrt ** nx1rt, x2ratrt ** nx2rt, x3ratrt ** nx3rt

        nx1mb, nx2mb, nx3mb = nx_meshblock
        x1minmb, x1maxmb, x1ratmb = x1f[:, 0], x1f[:, -1], x1ratrt ** (1 / (1 << levels))
        x2minmb, x2maxmb, x2ratmb = x2f[:, 0], x2f[:, -1], x2ratrt ** (1 / (1 << levels))
        x3minmb, x3maxmb, x3ratmb = x3f[:, 0], x3f[:, -1], x3ratrt ** (1 / (1 << levels))
        x1ratnxmb, x2ratnxmb, x3ratnxmb = x1ratmb ** nx1mb, x2ratmb ** nx2mb, x3ratmb ** nx3mb

        coordinates = self.header['Coordinates']
        ndim = int(np.sum(nx_root > 1))
        self.num_dimension = ndim

        (ix1, ox1), (ix2, ox2), (ix3, ox3) = self.boundaries

        # detect the number of ghost zone, assume the first meshblock has the logical location (0, 0, 0)
        ngh = np.searchsorted(x1v[0], x1minrt)
        self.num_ghost = ngh

        # calculate the numbers of finest meshblock needed
        nmb = [nx_root[d] // (nx_meshblock[d] - 2 * ngh) << maxlevel if nx_root[d] > 1 else 1 for d in range(3)]

        # assign meshblock ids to table
        mbtable = np.empty(nmb, dtype=int)
        for mb in range(self.header['NumMeshBlocks']):
            rngs = tuple(slice(llocs[mb, d] << levels[mb], llocs[mb, d] + 1 << levels[mb]) if nx_root[d] > 1 else 0
                         for d in range(3))
            mbtable[tuple(rngs)] = mb

        # start preparing functions...

        # given mesh position, return fractional position in root grid
        @nb.njit(fastmath=True)
        def mesh_position_to_fractional_position_root(x1_, x2_, x3_):
            if x1ratrt == 1.0:
                frac1_ = (x1_ - x1minrt) / (x1maxrt - x1minrt)
            else:
                frac1_ = np.log2(1 - (x1_ - x1minrt) / (x1maxrt - x1minrt) * (1 - x1ratnxrt)) / np.log2(x1ratnxrt)
            if x2ratrt == 1.0:
                frac2_ = (x2_ - x2minrt) / (x2maxrt - x2minrt)
            else:
                frac2_ = np.log2(1 - (x2_ - x2minrt) / (x2maxrt - x2minrt) * (1 - x2ratnxrt)) / np.log2(x2ratnxrt)
            if x3ratrt == 1.0:
                frac3_ = (x3_ - x3minrt) / (x3maxrt - x3minrt)
            else:
                frac3_ = np.log2(1 - (x3_ - x3minrt) / (x3maxrt - x3minrt) * (1 - x3ratnxrt)) / np.log2(x3ratnxrt)
            return frac1_, frac2_, frac3_
        self.mesh_position_to_fractional_position_root = mesh_position_to_fractional_position_root

        # given mesh position, return fractional position in meshblock
        @nb.njit(fastmath=True)
        def mesh_position_to_fractional_position_meshblock(mb_, x1_, x2_, x3_):
            x1min_, x1max_, x1rat_ = x1minmb[mb_], x1maxmb[mb_], x1ratmb[mb_]
            x2min_, x2max_, x2rat_ = x2minmb[mb_], x2maxmb[mb_], x2ratmb[mb_]
            x3min_, x3max_, x3rat_ = x3minmb[mb_], x3maxmb[mb_], x3ratmb[mb_]
            x1ratnx_, x2ratnx_, x3ratnx_ = x1ratnxmb[mb_], x2ratnxmb[mb_], x3ratnxmb[mb_]
            if x1rat_ == 1.0:
                frac1_ = (x1_ - x1min_) / (x1max_ - x1min_)
            else:
                frac1_ = np.log2(1 - (x1_ - x1min_) / (x1max_ - x1min_) * (1 - x1ratnx_)) / np.log2(x1ratnx_)
            if x2rat_ == 1.0:
                frac2_ = (x2_ - x2min_) / (x2max_ - x2min_)
            else:
                frac2_ = np.log2(1 - (x2_ - x2min_) / (x2max_ - x2min_) * (1 - x1ratnx_)) / np.log2(x1ratnx_)
            if x3rat_ == 1.0:
                frac3_ = (x3_ - x3min_) / (x3max_ - x3min_)
            else:
                frac3_ = np.log2(1 - (x3_ - x3min_) / (x3max_ - x3min_) * (1 - x3ratnx_)) / np.log2(x3ratnx_)
            return frac1_, frac2_, frac3_
        self.mesh_position_to_fractional_position_meshblock = mesh_position_to_fractional_position_meshblock

        # given mesh position, return meshblock id of the meshblock that contains the position
        @nb.njit(fastmath=True)
        def mesh_position_to_meshblock_id(x1_, x2_, x3_):
            frac1_, frac2_, frac3_ = mesh_position_to_fractional_position_root(x1_, x2_, x3_)
            if frac1_ < 0 or frac1_ >= 1 or frac2_ < 0 or frac2_ >= 1 or frac3_ < 0 or frac3_ >= 1:
                return -1
            mb1_ = int(frac1_ * mbtable.shape[0])
            mb2_ = int(frac2_ * mbtable.shape[1])
            mb3_ = int(frac3_ * mbtable.shape[2])
            return mbtable[mb1_, mb2_, mb3_]
        self.mesh_position_to_meshblock_id = mesh_position_to_meshblock_id

        # given mesh position, return indices in root grid
        @nb.njit(fastmath=True)
        def mesh_position_to_global_indices(x1_, x2_, x3_):
            frac1_, frac2_, frac3_ = mesh_position_to_fractional_position_root(x1_, x2_, x3_)
            gidx1_ = frac1_ * nx1rt
            gidx2_ = frac2_ * nx2rt
            gidx3_ = frac3_ * nx3rt
            return gidx1_, gidx2_, gidx3_
        self.mesh_position_to_global_indices = mesh_position_to_global_indices

        # given mesh position, return indices in root grid
        @nb.njit(fastmath=True)
        def mesh_position_to_local_indices(mb_, x1_, x2_, x3_):
            frac1_, frac2_, frac3_ = mesh_position_to_fractional_position_meshblock(mb_, x1_, x2_, x3_)
            lidx1_ = frac1_ * nx1mb
            lidx2_ = frac2_ * nx2mb
            lidx3_ = frac3_ * nx3mb
            return lidx1_, lidx2_, lidx3_
        self.mesh_position_to_local_indices = mesh_position_to_local_indices

        # given indices in root grid, return mesh position
        @nb.njit(fastmath=True)
        def global_indices_to_mesh_position(gidx1_, gidx2_, gidx3_):
            if x1ratrt == 1.0:
                x1_ = x1minrt + (x1maxrt - x1minrt) * gidx1_ / nx1rt
            else:
                x1_ = x1minrt + (x1maxrt - x1minrt) * (1 - x1ratrt ** gidx1_) / (1 - x1ratnxrt)
            if x2ratrt == 1.0:
                x2_ = x2minrt + (x2maxrt - x2minrt) * gidx2_ / nx2rt
            else:
                x2_ = x2minrt + (x2maxrt - x2minrt) * (1 - x2ratrt ** gidx2_) / (1 - x2ratnxrt)
            if x3ratrt == 1.0:
                x3_ = x3minrt + (x3maxrt - x3minrt) * gidx3_ / nx3rt
            else:
                x3_ = x3minrt + (x3maxrt - x3minrt) * (1 - x3ratrt ** gidx3_) / (1 - x3ratnxrt)
            return x1_, x2_, x3_
        self.global_indices_to_mesh_position = global_indices_to_mesh_position

        # given mesh positions and velocities, return derivatives in mesh position
        @nb.njit(fastmath=True)
        def velocity_to_derivatives(x1_, x2_, _x3_, v1_, v2_, v3_):
            if coordinates == 'cartesian':
                dx1_ = v1_
                dx2_ = v2_
                dx3_ = v3_
            elif coordinates == 'cylindrical':
                dx1_ = v1_
                dx2_ = v2_ / x1_
                dx3_ = v3_
            elif coordinates == 'spherical_polar':
                dx1_ = v1_
                dx2_ = v2_ / x1_
                dx3_ = v3_ / (x1_ * np.sin(x2_))
            else:
                raise RuntimeError('Unrecognized coordinates: ' + coordinates)
            return dx1_, dx2_, dx3_
        self.velocity_to_derivatives = velocity_to_derivatives

        # given mesh position, return the interpolated cell-centered quantities
        @nb.njit(fastmath=True)
        def interpolate_cell_centered(quantities_, mb_, x1_, x2_, x3_):
            if ndim == 1:
                w_ = np.ones((1, 1, 2), dtype=np.float64)
            elif ndim == 2:
                w_ = np.ones((1, 2, 2), dtype=np.float64)
            elif ndim == 3:
                w_ = np.ones((2, 2, 2), dtype=np.float64)
            else:
                raise RuntimeError('Unrecognized number of dimension: ' + str(ndim))

            lidx1_, lidx2_, lidx3_ = mesh_position_to_local_indices(mb_, x1_, x2_, x3_)

            l1s_ = int(lidx1_)
            if x1_ < x1v[mb_, l1s_]:
                w1_ = 0.5 + (x1_ - x1f[mb_, l1s_]) / (x1v[mb_, l1s_] - x1f[mb_, l1s_])
                l1s_, l1e_ = l1s_ - 1, l1s_ + 1
            else:
                w1_ = (x1_ - x1v[mb_, l1s_]) / (x1f[mb_, l1s_+1] - x1v[mb_, l1s_])
                l1s_, l1e_ = l1s_, l1s_ + 2
            w_[:, :, 0] *= 1.0 - w1_
            w_[:, :, 1] *= w1_

            if ndim >= 2:
                l2s_ = int(lidx2_)
                if x2_ < x2v[mb_, l2s_]:
                    w2_ = 0.5 + (x2_ - x2f[mb_, l2s_]) / (x2v[mb_, l2s_] - x2f[mb_, l2s_])
                    l2s_, l2e_ = l2s_ - 1, l2s_ + 1
                else:
                    w2_ = (x2_ - x2v[mb_, l2s_]) / (x2f[mb_, l2s_+1] - x2v[mb_, l2s_])
                    l2s_, l2e_ = l2s_, l2s_ + 2
                w_[:, 0, :] *= 1.0 - w2_
                w_[:, 1, :] *= w2_
            else:
                l2s_, l2e_ = 0, 1

            if ndim >= 3:
                l3s_ = int(lidx3_)
                if x3_ < x3v[mb_, l3s_]:
                    w3_ = 0.5 + (x3_ - x3f[mb_, l3s_]) / (x3v[mb_, l3s_] - x3f[mb_, l3s_])
                    l3s_, l3e_ = l3s_ - 1, l3s_ + 1
                else:
                    w3_ = (x3_ - x3v[mb_, l3s_]) / (x3f[mb_, l3s_+1] - x3v[mb_, l3s_])
                    l3s_, l3e_ = l3s_, l3s_ + 2
                w_[0, :, :] *= 1.0 - w3_
                w_[1, :, :] *= w3_
            else:
                l3s_, l3e_ = 0, 1
            return (quantities_[..., mb_, l3s_:l3e_, l2s_:l2e_, l1s_:l1e_] * w_).sum(axis=-1).sum(axis=-1).sum(axis=-1)
        self.interpolate_cell_centered = interpolate_cell_centered

        # given mesh position, return mesh position after applying boundary condition
        @nb.njit(fastmath=True)
        def apply_boundaries(x1_, x2_, x3_):
            if x1_ < x1minrt:
                if ix1 == 'none' or ix1 == 'outflow':
                    pass
                elif ix1 == 'reflecting':
                    x1_ = 2 * x1minrt - x1_
                elif ix1 == 'periodic':
                    x1_ = x1_ + (x1maxrt - x1minrt)
                else:
                    raise RuntimeError('Unrecognized boundary ix1 = ' + ix1)
            if x1_ >= x1maxrt:
                if ox1 == 'none' or ox1 == 'outflow':
                    pass
                elif ox1 == 'reflecting':
                    x1_ = 2 * x1maxrt - x1_
                elif ox1 == 'periodic':
                    x1_ = x1_ - (x1maxrt - x1minrt)
                else:
                    raise RuntimeError('Unrecognized boundary ox1 = ' + ox1)
            if x2_ < x2minrt:
                if ix2 == 'none' or ix2 == 'outflow':
                    pass
                elif ix2 == 'reflecting':
                    x2_ = 2 * x2minrt - x2_
                elif ix2 == 'periodic':
                    x2_ = x2_ + (x2maxrt - x2minrt)
                elif ix2 == 'polar':
                    if x2minrt != 0.0:
                        raise RuntimeError('ix2 = polar but x2min = ' + str(x2minrt) + ' != 0.0')
                    x2_ = -x2_
                    dx3_ = np.pi % (x3maxrt - x3minrt)
                    x3_ = x3_ + dx3_
                else:
                    raise RuntimeError('Unrecognized boundary ix2 = ' + ix2)
            if x2_ >= x2maxrt:
                if ox2 == 'none' or ox2 == 'outflow':
                    pass
                elif ox2 == 'reflecting':
                    x2_ = 2 * x2maxrt - x2_
                elif ox2 == 'periodic':
                    x2_ = x2_ - (x2maxrt - x2minrt)
                elif ox2 == 'polar':
                    if x2maxrt != np.pi:
                        raise RuntimeError('ox2 = polar but x2max = ' + str(x2maxrt) + ' != pi')
                    x2_ = x2maxrt - x2_
                    dx3_ = np.pi % (x3maxrt - x3minrt)
                    x3_ = x3_ + dx3_
                else:
                    raise RuntimeError('Unrecognized boundary ox2 = ' + ox2)
            if x3_ < x3minrt:
                if ix3 == 'none' or ix3 == 'outflow':
                    pass
                elif ix3 == 'reflecting':
                    x3_ = 2 * x3minrt - x3_
                elif ix3 == 'periodic':
                    x3_ = x3_ + (x3maxrt - x3minrt)
                else:
                    raise RuntimeError('Unrecognized boundary ix3 = ' + ix3)
            if x3_ >= x3maxrt:
                if ox3 == 'none' or ox3 == 'outflow':
                    pass
                elif ox3 == 'reflecting':
                    x3_ = 2 * x3maxrt - x3_
                elif ox3 == 'periodic':
                    x3_ = x3_ - (x3maxrt - x3minrt)
                else:
                    raise RuntimeError('Unrecognized boundary ox3 = ' + ox3)
            return x1_, x2_, x3_
        self.apply_boundaries = apply_boundaries
