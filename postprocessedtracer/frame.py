import itertools as it
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import numba as nb
import numpy as np


@dataclass(init=False, order=True)
class Frame:
    filename: str = field(compare=False)
    time: float = field(repr=False)
    header: dict = field(compare=False, repr=False)
    data: dict = field(compare=False, repr=False)
    derived_requirements: dict = field(compare=False, repr=False)
    num_ghost: int = field(compare=False, repr=False)
    num_dimension: int = field(compare=False, repr=False)
    boundaries: Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]] = field(compare=False, repr=False)
    boundary_str2func: dict = field(compare=False, repr=False)

    mesh_position_to_fractional_position_root: Callable = field(compare=False, repr=False)
    mesh_position_to_fractional_position_meshblock: Callable = field(compare=False, repr=False)
    mesh_position_to_meshblock_id: Callable = field(compare=False, repr=False)
    mesh_position_to_global_indices: Callable = field(compare=False, repr=False)
    mesh_position_to_local_indices: Callable = field(compare=False, repr=False)
    global_indices_to_mesh_position: Callable = field(compare=False, repr=False)
    velocity_to_derivatives: Callable = field(compare=False, repr=False)
    interpolate_cell_centered: Callable = field(compare=False, repr=False)
    apply_boundaries: Callable = field(compare=False, repr=False)
    get_finite_volume: Callable = field(compare=False, repr=False)

    def __init__(self, filename, boundaries=None, num_ghost=None):
        self.filename = filename

        if boundaries is None:
            self.boundaries = (('none',) * 2,) * 3
        else:
            try:
                (ix1, ox1), (ix2, ox2), (ix3, ox3) = boundaries
                self.boundaries = ((str(ix1), str(ox1)), (str(ix2), str(ox2)), (str(ix3), str(ox3)))
            except ValueError:
                raise ValueError('boundaries has to be in the form: ((ix1, ox1), (ix2, ox2), (ix3, ox3))')

        self._load_header()
        self.time = self.header['Time']
        self.data = {}

        self._set_ghostzone(num_ghost)

        self.derived_requirements = {
            'int_rho_dr': ['rho']
        }

        self.boundary_str2func = defaultdict(lambda: {
            'none': defaultdict(lambda: None),
            'outflow': defaultdict(lambda: _outflow_boundary),
            'reflecting': defaultdict(lambda: _reflect_boundary),
            'periodic': defaultdict(lambda: None),
            'polar': defaultdict(lambda: _reflect_boundary),
        })

        self.boundary_str2func['vel1']['reflecting']['ix1'] = _negative_reflect_boundary
        self.boundary_str2func['vel1']['reflecting']['ox1'] = _negative_reflect_boundary
        self.boundary_str2func['vel2']['reflecting']['ix2'] = _negative_reflect_boundary
        self.boundary_str2func['vel2']['reflecting']['ox2'] = _negative_reflect_boundary
        self.boundary_str2func['vel3']['reflecting']['ix3'] = _negative_reflect_boundary
        self.boundary_str2func['vel3']['reflecting']['ox3'] = _negative_reflect_boundary
        self.boundary_str2func['vel2']['polar']['ix2'] = _negative_reflect_boundary
        self.boundary_str2func['vel2']['polar']['ox2'] = _negative_reflect_boundary
        self.boundary_str2func['vel3']['polar']['ix2'] = _negative_reflect_boundary
        self.boundary_str2func['vel3']['polar']['ox2'] = _negative_reflect_boundary

        self._prepare_functions()

    def load(self, quantities=None):
        import h5py

        if quantities is None:
            quantities = list(self.header['VariableNames'])

        num_variables = self.header['NumVariables']
        dataset_prefix = np.cumsum(num_variables)
        dataset_offsets = dataset_prefix - dataset_prefix[0]

        derived_quantities = []

        with h5py.File(self.filename, 'r') as f:
            dataset_names = self.header['DatasetNames']
            variable_names = self.header['VariableNames']

            for q in quantities:
                if q in self.data:
                    continue

                if q not in variable_names and q not in self.derived_requirements:
                    raise RuntimeError(f'Quantity "{q}" not found, available quantities include '
                                       f'{variable_names + list(self.derived_requirements.keys())}')

                if q in variable_names:
                    variable_index = variable_names.index(q)
                    dataset_index = np.searchsorted(dataset_prefix, variable_index)
                    variable_index -= dataset_offsets[dataset_index]
                    self.data[q] = _convert_type(f[dataset_names[dataset_index]][variable_index])
                    if self.num_ghost < self.num_ghost_data:
                        # simply crop the data
                        diff = self.num_ghost_data - self.num_ghost
                        slc = tuple(slice(None) if i == 0 or n == 1 else slice(diff, n - diff) for i, n in enumerate(self.data[q].shape))
                        self.data[q] = self.data[q][slc]
                    elif self.num_ghost > self.num_ghost_data:
                        # pad and fix boundary
                        diff = self.num_ghost - self.num_ghost_data
                        pad = tuple((0, 0) if i == 0 or n == 1 else (diff, diff) for i, n in enumerate(self.data[q].shape))
                        self.data[q] = np.pad(self.data[q], pad)
                    (ix1, ox1), (ix2, ox2), (ix3, ox3) = self.boundaries
                    ix1 = self.boundary_str2func[q][ix1]['ix1']
                    ox1 = self.boundary_str2func[q][ox1]['ox1']
                    ix2 = self.boundary_str2func[q][ix2]['ix2']
                    ox2 = self.boundary_str2func[q][ox2]['ox2']
                    ix3 = self.boundary_str2func[q][ix3]['ix3']
                    ox3 = self.boundary_str2func[q][ox3]['ox3']
                    _fix_boundary(self, q, boundary_func=((ix1, ox1), (ix2, ox2), (ix3, ox3)))
                elif q in self.derived_requirements:
                    quantities.extend(self.derived_requirements[q])
                    derived_quantities.append(q)

        for q in derived_quantities:
            if q in self.data:
                continue
            self.derive_quantity(q)

    def unload(self):
        self.data = {}

    def derive_quantity(self, quantity):
        if quantity == 'int_rho_dr':
            self.data[quantity] = _integrate_x1(self, 'rho')
            # self.patch_boundary(['int_rho_dr'])
            _fix_boundary(self, 'int_rho_dr', boundary_func=((_outflow_boundary, _negative_reflect_boundary),
                                                             (_reflect_boundary, _reflect_boundary), (None, None)))

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

    def _set_ghostzone(self, num_ghost):
        # detect the number of ghost zone, assume the first meshblock has the logical location (0, 0, 0)
        x1v = self.header['x1v']
        x1minrt, x1maxrt, x1ratrt = self.header['RootGridX1']
        self.num_ghost_data = num_ghost_data = np.searchsorted(x1v[0], x1minrt)
        if num_ghost is None:
            self.num_ghost = num_ghost = self.num_ghost_data
        else:
            self.num_ghost = num_ghost

        if num_ghost == num_ghost_data:
            # no action needed
            return

        if num_ghost < num_ghost_data:
            # simply crop the data
            diff = num_ghost_data - num_ghost
            for i in range(1, 4):
                if self.header['MeshBlockSize'][i-1] == 1:
                    continue
                self.header['MeshBlockSize'][i-1] -= diff * 2
                for vf in 'vf':
                    xi = f'x{i}{vf}'
                    self.header[xi] = self.header[xi][:, diff:-diff]
            return

        # extend all data
        # tricky part: extending cell centers is not trivial
        # try xv[i] == (a-1)/a * ((xf[i+1]**a - xf[i]**a) / (xf[i+1]**(a-1) - xf[i]**(a-1)))
        # with a = 2, 3, 4 (cartesian, cylindrical, spherical polar)
        # also try ((sin(xf[i+1]) - xf[i+1] cos(xf[i+1])) - (sin(xf[i]) - xf[i] cos(xf[i])))/(cos(xf[i]) - cos(xf[i+1]))
        diff = num_ghost - num_ghost_data
        for i in range(1, 4):
            if self.header['MeshBlockSize'][i-1] == 1:
                continue
            self.header['MeshBlockSize'][i-1] += diff * 2
            for vf in 'vf':
                xi = f'x{i}{vf}'
                self.header[xi] = np.pad(self.header[xi], ((0, 0), (diff, diff)))
            xif = self.header[f'x{i}f']
            xiv = self.header[f'x{i}v']
            for a in range(2, 5):
                if np.isclose((a-1)/a * ((xif[0, diff+1]**a - xif[0, diff]**a)/(xif[0, diff+1]**(a-1) - xif[0, diff]**(a-1))), xiv[0, diff], rtol=1e-15):
                    break
            else:
                if np.isclose(((np.sin(xif[0, diff+1]) - xif[0, diff+1] * np.cos(xif[0, diff+1])) - (np.sin(xif[0, diff]) - xif[0, diff] * np.cos(xif[0, diff])))
                              / (np.cos(xif[0, diff]) - np.cos(xif[0, diff+1])), xiv[0, diff], rtol=1e-15):
                    a = 0
                else:
                    raise RuntimeError(f'Can\'t extend cell centers in the x{i}-direction')
            levels = self.header['Levels']
            rat = self.header[f'RootGridX{i}'][2]**(1 / (1 << levels))
            for j in range(diff, 0, -1):
                xif[:, j-1] = xif[:, j] - (xif[:, j+1] - xif[:, j]) / rat
                if a > 0:
                    xiv[:, j-1] = (a-1)/a * ((xif[:, j]**a - xif[:, j-1]**a)/(xif[:, j]**(a-1) - xif[:, j-1]**(a-1)))
                else:
                    xiv[:, j-1] = ((np.sin(xif[:, j]) - xif[:, j] * np.cos(xif[:, j])) - (np.sin(xif[:, j-1]) - xif[:, j-1] * np.cos(xif[:, j-1]))
                                   / (np.cos(xif[:, j-1]) - np.cos(xif[:, j])))
                xif[:, -j] = xif[:, -j-1] + (xif[:, -j-1] - xif[:, -j-2]) * rat
                if a > 0:
                    xiv[:, -j] = (a-1)/a * ((xif[:, -j]**a - xif[:, -j-1]**a)/(xif[:, -j]**(a-1) - xif[:, -j-1]**(a-1)))
                else:
                    xiv[:, -j] = ((np.sin(xif[:, -j]) - xif[:, -j] * np.cos(xif[:, -j])) - (np.sin(xif[:, -j-1]) - xif[:, -j-1] * np.cos(xif[:, -j-1]))
                                  / (np.cos(xif[:, -j-1]) - np.cos(xif[:, -j])))

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

        # calculate the numbers of finest meshblock needed
        ngh = self.num_ghost
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
            mb1_ = min(max(0, int(frac1_ * mbtable.shape[0])), mbtable.shape[0] - 1)
            mb2_ = min(max(0, int(frac2_ * mbtable.shape[1])), mbtable.shape[1] - 1)
            mb3_ = min(max(0, int(frac3_ * mbtable.shape[2])), mbtable.shape[2] - 1)
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
                w1_ = 0.5 + 0.5 * (x1_ - x1f[mb_, l1s_]) / (x1v[mb_, l1s_] - x1f[mb_, l1s_])
                l1s_, l1e_ = l1s_ - 1, l1s_ + 1
            else:
                w1_ = 0.5 * (x1_ - x1v[mb_, l1s_]) / (x1f[mb_, l1s_ + 1] - x1v[mb_, l1s_])
                l1s_, l1e_ = l1s_, l1s_ + 2
            w_[:, :, 0] *= 1.0 - w1_
            w_[:, :, 1] *= w1_

            if ndim >= 2:
                l2s_ = int(lidx2_)
                if x2_ < x2v[mb_, l2s_]:
                    w2_ = 0.5 + 0.5 * (x2_ - x2f[mb_, l2s_]) / (x2v[mb_, l2s_] - x2f[mb_, l2s_])
                    l2s_, l2e_ = l2s_ - 1, l2s_ + 1
                else:
                    w2_ = 0.5 * (x2_ - x2v[mb_, l2s_]) / (x2f[mb_, l2s_ + 1] - x2v[mb_, l2s_])
                    l2s_, l2e_ = l2s_, l2s_ + 2
                w_[:, 0, :] *= 1.0 - w2_
                w_[:, 1, :] *= w2_
            else:
                l2s_, l2e_ = 0, 1

            if ndim >= 3:
                l3s_ = int(lidx3_)
                if x3_ < x3v[mb_, l3s_]:
                    w3_ = 0.5 + 0.5 * (x3_ - x3f[mb_, l3s_]) / (x3v[mb_, l3s_] - x3f[mb_, l3s_])
                    l3s_, l3e_ = l3s_ - 1, l3s_ + 1
                else:
                    w3_ = 0.5 * (x3_ - x3v[mb_, l3s_]) / (x3f[mb_, l3s_ + 1] - x3v[mb_, l3s_])
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
            if x1_ > x1maxrt:
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
            if x2_ > x2maxrt:
                if ox2 == 'none' or ox2 == 'outflow':
                    pass
                elif ox2 == 'reflecting':
                    x2_ = 2 * x2maxrt - x2_
                elif ox2 == 'periodic':
                    x2_ = x2_ - (x2maxrt - x2minrt)
                elif ox2 == 'polar':
                    # if x2maxrt != np.pi:
                    #     raise RuntimeError('ox2 = polar but x2max = ' + str(x2maxrt) + ' != pi')
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
            if x3_ > x3maxrt:
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

        # return finite volume
        def get_finite_volume():
            if coordinates == 'cartesian':
                dx1 = np.diff(self.header['x1f'], axis=-1)
                dx2 = np.diff(self.header['x2f'], axis=-1)
                dx3 = np.diff(self.header['x3f'], axis=-1)
            elif coordinates == 'cylindrical':
                dx1 = np.diff(self.header['x1f'] ** 2 / 2, axis=-1)
                dx2 = np.diff(self.header['x2f'], axis=-1)
                dx3 = np.diff(self.header['x3f'], axis=-1)
            elif coordinates == 'spherical_polar':
                dx1 = np.diff(self.header['x1f'] ** 3 / 3, axis=-1)
                dx2 = np.diff(-np.cos(self.header['x2f']), axis=-1)
                dx3 = np.diff(self.header['x3f'], axis=-1)
            else:
                raise RuntimeError('Unrecognized coordinates: ' + coordinates)
            dvol = dx1[:, None, None, :] * dx2[:, None, :, None] * dx3[:, :, None, None]
            return dvol

        self.get_finite_volume = get_finite_volume

    def patch_boundary(self, quantities):
        lloc2mb = dict()
        for mb, lloc in enumerate(self.header['LogicalLocations']):
            lloc2mb[tuple(lloc)] = mb

        def patch_one(data):
            nx = self.header['MeshBlockSize']
            ngh = self.num_ghost
            ndim = self.num_dimension
            xrngs = [(-1, (0, ngh)), (0, (ngh, nx[0] - ngh)), (1, (nx[0] - ngh, nx[0]))]
            yrngs = [(-1, (0, ngh)), (0, (ngh, nx[1] - ngh)), (1, (nx[1] - ngh, nx[1]))] \
                if ndim >= 2 else [(0, (0, nx[1]))]
            zrngs = [(-1, (0, ngh)), (0, (ngh, nx[2] - ngh)), (1, (nx[2] - ngh, nx[2]))] \
                if ndim >= 3 else [(0, (0, nx[2]))]
            nx0 = nx[0] - 2 * ngh
            nx1 = nx[1] - 2 * ngh if ndim >= 2 else nx[1]
            nx2 = nx[2] - 2 * ngh if ndim >= 3 else nx[2]
            for mymb, mylloc in enumerate(self.header['LogicalLocations']):
                for z, (zs, ze) in zrngs:
                    for y, (ys, ye) in yrngs:
                        for x, (xs, xe) in xrngs:
                            if x == y == z == 0:
                                continue
                            nblloc = mylloc + np.array([x, y, z])
                            if tuple(nblloc) not in lloc2mb:
                                continue
                            nbmb = lloc2mb[tuple(nblloc)]
                            data[mymb, zs:ze, ys:ye, xs:xe] = \
                                data[nbmb, zs-z*nx2:ze-z*nx2, ys-y*nx1:ye-y*nx1, xs-x*nx0:xe-x*nx0]
        for q in quantities:
            patch_one(self.data[q])

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


def _integrate_x1(frame: Frame, quantity):
    class PartialIntegralTree:
        level: int
        lloc: np.ndarray
        grid_size: np.ndarray
        node_value: np.ndarray
        leaf_value: np.ndarray
        leaf: List['PartialIntegralTree']

        def __init__(self, level: int, lloc: np.ndarray, grid_size: np.ndarray):
            self.level = level
            self.lloc = lloc
            self.grid_size = grid_size
            self.node_value = np.zeros(grid_size, dtype=float)
            self.leaf_value = np.zeros(grid_size, dtype=float)
            self.leaf = [None] * 4

        def add(self, level: int, lloc: np.ndarray, value: np.ndarray):
            self._add_helper(level, lloc, value)

        def _add_helper(self, level: int, lloc: np.ndarray, value: np.ndarray) -> Tuple:
            if level == self.level:
                self.node_value += value
                return np.array(value), np.array([0, 0])

            leaf, lc = self.get_leaf(level, lloc)
            coarse_value, jks = leaf._add_helper(level, lloc, value)
            if coarse_value.shape[0] > 1:
                coarse_value[0::2, :] += coarse_value[1::2, :]
            if coarse_value.shape[1] > 1:
                coarse_value[:, 0::2] += coarse_value[:, 1::2]
            coarse_value = coarse_value[::2, ::2]
            coarse_value *= 0.5 ** sum(self.grid_size > 1)
            js, ks = jks = (jks + lc * self.grid_size) >> 1
            je, ke = jks + np.array(coarse_value.shape)
            self.leaf_value[js:je, ks:ke] += coarse_value

            return coarse_value, jks

        def get(self, level: int, lloc: np.ndarray) -> np.ndarray:
            return self._get_helper(level, lloc)[0]

        def _get_helper(self, level: int, lloc: np.ndarray):
            if level == self.level:
                return self.node_value + self.leaf_value, np.array([0, 0]), self.grid_size.copy()

            leaf, lc = self.get_leaf(level, lloc)
            result, jks, sz = leaf._get_helper(level, lloc)
            if sz[0] > 1:
                sz[0] >>= 1
            if sz[1] > 1:
                sz[1] >>= 1
            js, ks = jks = (jks + lc * self.grid_size) >> 1
            je, ke = jks + sz
            result += np.repeat(np.repeat(self.node_value[js:je, ks:ke], self.grid_size[0] // sz[0], axis=0),
                                self.grid_size[1] // sz[1], axis=1)

            return result, jks, sz

        def get_leaf(self, level: int, lloc: np.ndarray) -> Tuple['PartialIntegralTree', np.ndarray]:
            lc = lloc >> (level - self.level - 1) & 1
            i = lc[0] + lc[1] * 2
            if self.leaf[i] is None:
                self.leaf[i] = PartialIntegralTree(self.level + 1, lloc * 2 + lc, self.grid_size)
            return self.leaf[i], lc

        @staticmethod
        def from_frame(frame_: Frame):
            locmax = np.max(frame_.header['RootGridSize'][1:frame_.num_dimension] / (
                    frame_.header['MeshBlockSize'][1:frame_.num_dimension] - 2 * frame_.num_ghost))
            level, size = 0, 1
            while size < locmax:
                level += 1
                size *= 2
            grid_size = frame_.header['MeshBlockSize'][1:].copy()
            grid_size[:frame_.num_dimension - 1] -= 2 * frame_.num_ghost
            return PartialIntegralTree(-level, np.array([0, 0]), grid_size)

    frame.load([quantity])
    ng = frame.num_ghost

    slcs = (slice(None),) * (3 - frame.num_dimension) + (slice(ng, -ng),) * frame.num_dimension
    tree = PartialIntegralTree.from_frame(frame)
    res = np.zeros_like(frame.data[quantity])
    for mb in sorted(range(frame.header['NumMeshBlocks']), key=lambda i: frame.header['LogicalLocations'][i, 0] << (
            frame.header['MaxLevel'] - frame.header['Levels'][i]), reverse=True):
        dx1f = np.diff(frame.header['x1f'][mb, ng:-ng])[None, None, :]
        dx1vf = (frame.header['x1v'][mb, ng:-ng] - frame.header['x1f'][mb, ng:-ng - 1])[None, None, :]
        rho = frame.data[quantity][(mb, *slcs)]
        dtau = np.cumsum((rho * dx1f)[:, :, ::-1], axis=2)[:, :, ::-1]
        start = tree.get(frame.header['Levels'][mb], frame.header['LogicalLocations'][mb, 1:])
        tree.add(frame.header['Levels'][mb], frame.header['LogicalLocations'][mb, 1:], dtau[:, :, 0].T)
        res[(mb, *slcs)] = start.T[:, :, None] + dtau - rho * dx1vf
    return res


def _fix_boundary(frame: Frame, quantity: str, boundary_func):
    mbfinder = dict()
    for mb in range(frame.header['NumMeshBlocks']):
        level = frame.header['Levels'][mb]
        lloc = frame.header['LogicalLocations'][mb]
        mbfinder[(level, tuple(lloc))] = mb

    arr = frame.data[quantity]
    for mymb in range(frame.header['NumMeshBlocks']):
        ngh = frame.num_ghost
        mylevel = frame.header['Levels'][mymb]
        mylloc = frame.header['LogicalLocations'][mymb]
        active = [True] * frame.num_dimension + [False] * (3 - frame.num_dimension)
        xls = [-1, 0, 1]
        yls = [-1, 0, 1] if active[1] else [0]
        zls = [-1, 0, 1] if active[2] else [0]
        il, iu = (ngh, frame.header['MeshBlockSize'][0] - ngh)
        jl, ju = (ngh, frame.header['MeshBlockSize'][1] - ngh) if active[1] else (0, 1)
        kl, ku = (ngh, frame.header['MeshBlockSize'][2] - ngh) if active[2] else (0, 1)
        nx1, nx2, nx3 = iu - il, ju - jl, ku - kl
        for x, y, z in sorted(it.product(xls, yls, zls), key=lambda xyz_: sum(map(abs, xyz_))):
            if x == y == z == 0:
                continue
            myil, myiu = (0, il) if x == -1 else (iu, iu + ngh) if x == 1 else (il, iu)
            myjl, myju = (0, jl) if y == -1 else (ju, ju + ngh) if y == 1 else (jl, ju)
            mykl, myku = (0, kl) if z == -1 else (ku, ku + ngh) if z == 1 else (kl, ku)
            boundary = None

            # first, find if there is meshblock at same level
            theirlevel = mylevel
            theirlloc = mylloc + np.array([x, y, z])
            if (theirlevel, tuple(theirlloc)) in mbfinder:
                theirmb = mbfinder[(theirlevel, tuple(theirlloc))]
                theiril, theiriu = (iu - ngh, iu) if x == -1 else (ngh, ngh + ngh) if x == 1 else (il, iu)
                theirjl, theirju = (ju - ngh, ju) if y == -1 else (ngh, ngh + ngh) if y == 1 else (jl, ju)
                theirkl, theirku = (ku - ngh, ku) if z == -1 else (ngh, ngh + ngh) if z == 1 else (kl, ku)
                boundary = arr[theirmb, theirkl:theirku, theirjl:theirju, theiril:theiriu]

            # second, find if there is coarser meshblock
            theirlevel = mylevel - 1
            theirlloc = (mylloc + np.array([x, y, z])) >> 1
            if boundary is None and (theirlevel, tuple(theirlloc)) in mbfinder:
                theirmb = mbfinder[(theirlevel, tuple(theirlloc))]
                theiril, theiriu = ((iu - (ngh >> 1), iu) if x == -1 else (ngh, ngh + (ngh >> 1)) if x == 1 else
                (il + (nx1 >> 1), iu) if mylloc[0] & 1 else (il, iu - (nx1 >> 1)))
                theirjl, theirju = ((ju - (ngh >> 1), ju) if y == -1 else (ngh, ngh + (ngh >> 1)) if y == 1 else
                (jl + (nx2 >> 1), ju) if mylloc[1] & 1 else (jl, ju - (nx2 >> 1)))
                theirkl, theirku = ((ku - (ngh >> 1), ku) if z == -1 else (ngh, ngh + (ngh >> 1)) if z == 1 else
                (kl + (nx3 >> 1), ku) if mylloc[2] & 1 else (kl, ku - (nx3 >> 1)))
                boundary = np.repeat(np.repeat(np.repeat(
                    arr[theirmb, theirkl:theirku, theirjl:theirju, theiril:theiriu], 2, axis=0), 2, axis=1), 2, axis=3)

            # third, find if there is finer meshblock
            theirlevel = mylevel + 1
            theirlloc = (mylloc + np.array([x, y, z])) << 1
            if boundary is None and (theirlevel, tuple(theirlloc)) in mbfinder:
                theiril, theiriu = (iu - ngh, iu) if x == -1 else (ngh, ngh + ngh) if x == 1 else (il, iu)
                theirjl, theirju = (ju - ngh, ju) if y == -1 else (ngh, ngh + ngh) if y == 1 else (jl, ju)
                theirkl, theirku = (ku - ngh, ku) if z == -1 else (ngh, ngh + ngh) if z == 1 else (kl, ku)

                boundary = []
                for z_ in range(active[2] + 1):
                    boundary_ = []
                    for y_ in range(active[1] + 1):
                        boundary__ = []
                        for x_ in range(active[0] + 1):
                            theirmb = mbfinder[(theirlevel, tuple(theirlloc + np.array([x_, y_, z_])))]
                            boundary__.append(arr[theirmb, theirkl:theirku, theirjl:theirju, theiril:theiriu])
                        boundary_.append(boundary__)
                    boundary.append(boundary_)
                boundary = np.block(boundary)
                boundary[:, :, 0::2] += boundary[:, :, 1::2]
                if active[1]:
                    boundary[:, 0::2, 0::2] += boundary[:, 1::2, 0::2]
                if active[2]:
                    boundary[0::2, 0::2, 0::2] += boundary[1::2, 0::2, 0::2]
                boundary *= 0.5 ** sum(active)

            if boundary is None:
                meshblock_size = frame.header['MeshBlockSize'].copy()
                meshblock_size[:frame.num_dimension] -= ngh * 2
                root_indices = (theirlloc >> theirlevel) * meshblock_size
                for i, (idx, sz) in enumerate(zip(root_indices, frame.header['RootGridSize'])):
                    d = -1 if idx < 0 else 1 if idx >= sz else 0
                    if d != 0:
                        func = boundary_func[i][(d + 1) // 2]
                        if func is None:
                            continue
                        arr_ = arr[mymb]
                        arr_ = np.moveaxis(arr_, 2 - i, 2)
                        if d == 1:
                            arr_ = np.flip(arr_, 2)
                        jl_, ju_ = (myjl, myju) if i == 0 else (myil, myiu)
                        kl_, ku_ = (myjl, myju) if i == 2 else (mykl, myku)
                        boundary = func(arr_, ngh, jl_, ju_, kl_, ku_)
                        if d == 1:
                            boundary = np.flip(boundary, 2)
                        boundary = np.moveaxis(boundary, 2, 2 - i)
                        break
                else:
                    boundary = np.full((myku - mykl, myju - myjl, myiu - myil), np.nan)

            arr[mymb, mykl:myku, myjl:myju, myil:myiu] = boundary


def _zero_boundary(arr, ngh, jl, ju, kl, ku):
    return np.zeros_like(arr[kl:ku, jl:ju, 0:ngh])


def _outflow_boundary(arr, ngh, jl, ju, kl, ku):
    return np.repeat(arr[kl:ku, jl:ju, ngh:ngh + 1], ngh, axis=2)


def _reflect_boundary(arr, ngh, jl, ju, kl, ku):
    return arr[kl:ku, jl:ju, ngh:ngh + ngh][:, :, ::-1]


def _negative_reflect_boundary(arr, ngh, jl, ju, kl, ku):
    return -arr[kl:ku, jl:ju, ngh:ngh + ngh][:, :, ::-1]
