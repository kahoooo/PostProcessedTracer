import numba as nb
import numpy as np
from frame import Frame
from particles import Particles
import json


def _convert_to_serializable(x):
    if isinstance(x, np.bytes_) or isinstance(x, bytes):
        out_ = x.decode('ascii', 'replacec')
    elif isinstance(x, np.int32):
        out_ = int(x)
    elif isinstance(x, np.float64):
        out_ = float(x)
    elif isinstance(x, np.ndarray):
        out_ = _convert_to_serializable(x.tolist())
    elif isinstance(x, list):
        out_ = list(map(_convert_to_serializable, x))
    elif isinstance(x, dict):
        out_ = dict()
        for key, value in x.items():
            key = _convert_to_serializable(key)
            value = _convert_to_serializable(value)
            out_[key] = value
    else:
        out_ = x
    return out_


class Integrator:
    cfl: float
    cfl_inactive: float
    compute: dict

    def __init__(self, cfl: float = 0.3, cfl_inactive: float = np.inf):
        self.cfl = cfl
        self.cfl_inactive = cfl_inactive
        self.compute = dict()

    def _make_compute_function(self, first: Frame, second: Frame):
        raise RuntimeError('This function is supposed to be implemented by derived classes')

    def get_compute_function(self, first: Frame, second: Frame):
        h1 = dict(first.header)
        h2 = dict(second.header)
        for key in ['Time', 'NumCycles']:
            del h1[key], h2[key]

        hash_key = json.dumps(_convert_to_serializable([h1, h2, np.sign(second.time - first.time)]))
        if hash_key not in self.compute:
            self.compute[hash_key] = self._make_compute_function(first, second)
        return self.compute[hash_key]

    def integrate(self, first: Frame, second: Frame, par: Particles):
        if (np.any(first.header['RootGridX1'][:2] != second.header['RootGridX1'][:2])
                or np.any(first.header['RootGridX2'][:2] != second.header['RootGridX2'][:2])
                or np.any(first.header['RootGridX3'][:2] != second.header['RootGridX3'][:2])):
            raise RuntimeError('Domain sizes of the two frames are different')

        if first.time == second.time:
            return
        delta_t = abs(second.time - first.time)

        print('Integrating from ' + first.filename + ' to ' + second.filename)
        compute = self.get_compute_function(first, second)

        vel1 = np.stack([first.data[f'vel{d+1}'] for d in range(3)])
        vel2 = np.stack([second.data[f'vel{d+1}'] for d in range(3)])
        newsize = compute(delta_t, vel1, vel2, par.pids, par.meshs)
        par.resize(newsize)
        print(f'{newsize} particles remain in bound')


class VanLeer2(Integrator):
    def _make_compute_function(self, first: Frame, second: Frame):
        nx_root = first.header['RootGridSize']
        cfl = np.where(nx_root > 1, self.cfl, self.cfl_inactive)
        sign = np.sign(second.time - first.time)

        # variables valid for both frame
        x1minrt, x1maxrt, _ = first.header['RootGridX1']
        x2minrt, x2maxrt, _ = first.header['RootGridX2']
        x3minrt, x3maxrt, _ = first.header['RootGridX3']

        vel2derv = first.velocity_to_derivatives
        boundary = first.apply_boundaries

        # variables valid for first frame
        ngh1 = first.num_ghost
        ndim1 = first.num_dimension
        x1f1, x2f1, x3f1 = first.header['x1f'], first.header['x2f'], first.header['x3f']
        dx1f1, dx2f1, dx3f1 = np.diff(x1f1, axis=-1), np.diff(x2f1, axis=-1), np.diff(x3f1, axis=-1)
        x1minmb1, x1maxmb1 = x1f1[:, ngh1], x1f1[:, -ngh1]
        x2minmb1, x2maxmb1 = (x2f1[:, ngh1], x2f1[:, -ngh1]) if ndim1 >= 2 else (x2f1[:, 0], x2f1[:, 1])
        x3minmb1, x3maxmb1 = (x3f1[:, ngh1], x3f1[:, -ngh1]) if ndim1 >= 3 else (x3f1[:, 0], x3f1[:, 1])

        mesh2mb1 = first.mesh_position_to_meshblock_id
        mesh2lidx1 = first.mesh_position_to_local_indices
        interpcc1 = first.interpolate_cell_centered

        # variables valid for second frame
        ngh2 = second.num_ghost
        ndim2 = second.num_dimension
        x1f2, x2f2, x3f2 = second.header['x1f'], second.header['x2f'], second.header['x3f']
        dx1f2, dx2f2, dx3f2 = np.diff(x1f2, axis=-1), np.diff(x2f2, axis=-1), np.diff(x3f2, axis=-1)
        x1minmb2, x1maxmb2 = x1f2[:, ngh2], x1f2[:, -ngh2]
        x2minmb2, x2maxmb2 = (x2f2[:, ngh2], x2f2[:, -ngh2]) if ndim2 >= 2 else (x2f2[:, 0], x2f2[:, 1])
        x3minmb2, x3maxmb2 = (x3f2[:, ngh2], x3f2[:, -ngh2]) if ndim2 >= 3 else (x3f2[:, 0], x3f2[:, 1])

        mesh2mb2 = second.mesh_position_to_meshblock_id
        mesh2lidx2 = second.mesh_position_to_local_indices
        interpcc2 = second.interpolate_cell_centered

        @nb.njit(fastmath=True)
        def compute(delta_t_, vel1_, vel2_, pids_, meshs_):
            # during integration, particles are separated into three types: finished, out-of-bound and remaining
            # particles are ordered such that finished particles would be in the front, and out-of-bound particles would
            # be at the back, the remaining particles are ordered by (meshblock id1, meshblock id2, time elapsed)
            # after integration, particles would be out-of-order

            npar_ = pids_.size
            finished_ = 0
            in_bound_ = npar_
            remaining_ = npar_

            time_elapsed_ = np.zeros(npar_, dtype=np.float64)
            types_ = np.zeros(npar_, dtype=np.float64)
            mbs1_ = np.empty_like(pids_, dtype=np.int64)
            mbs2_ = np.empty_like(pids_, dtype=np.int64)

            while remaining_ > 0:
                # all remaining particles should be in-bound and not finished integrating
                # calculate logical locations and get meshblock ids
                for p_ in nb.prange(finished_, in_bound_):
                    mbs1_[p_] = mesh2mb1(meshs_[p_, 0], meshs_[p_, 1], meshs_[p_, 2])
                    mbs2_[p_] = mesh2mb2(meshs_[p_, 0], meshs_[p_, 1], meshs_[p_, 2])

                # first, sort particles by (meshblock id1, meshblock id2, time elapsed)
                # do stable sort three times, least significant first
                order_ = time_elapsed_[finished_:in_bound_].argsort(kind='mergesort')
                order_ += finished_
                pids_[finished_:in_bound_] = pids_[order_]
                meshs_[finished_:in_bound_, :] = meshs_[order_, :]
                time_elapsed_[finished_:in_bound_] = time_elapsed_[order_]
                mbs1_[finished_:in_bound_] = mbs1_[order_]
                mbs2_[finished_:in_bound_] = mbs2_[order_]
                order_ = mbs2_[finished_:in_bound_].argsort(kind='mergesort')
                order_ += finished_
                pids_[finished_:in_bound_] = pids_[order_]
                meshs_[finished_:in_bound_, :] = meshs_[order_, :]
                time_elapsed_[finished_:in_bound_] = time_elapsed_[order_]
                mbs1_[finished_:in_bound_] = mbs1_[order_]
                mbs2_[finished_:in_bound_] = mbs2_[order_]
                order_ = mbs1_[finished_:in_bound_].argsort(kind='mergesort')
                order_ += finished_
                pids_[finished_:in_bound_] = pids_[order_]
                meshs_[finished_:in_bound_, :] = meshs_[order_, :]
                time_elapsed_[finished_:in_bound_] = time_elapsed_[order_]
                mbs1_[finished_:in_bound_] = mbs1_[order_]
                mbs2_[finished_:in_bound_] = mbs2_[order_]

                for p_ in nb.prange(finished_, in_bound_):
                    mb1_ = mbs1_[p_]
                    mb2_ = mbs2_[p_]

                    # overlapped domain
                    x1min_, x1max_ = max(x1minmb1[mb1_], x1minmb2[mb2_]), min(x1maxmb1[mb1_], x1maxmb2[mb2_])
                    x2min_, x2max_ = max(x2minmb1[mb1_], x2minmb2[mb2_]), min(x2maxmb1[mb1_], x2maxmb2[mb2_])
                    x3min_, x3max_ = max(x3minmb1[mb1_], x3minmb2[mb2_]), min(x3maxmb1[mb1_], x3maxmb2[mb2_])

                    # temporary variables
                    time_ = time_elapsed_[p_]
                    dt_ = delta_t_ * 1e-6
                    x1_, x2_, x3_ = meshs_[p_]

                    while time_ < delta_t_:
                        # get velocity and transform to derivatives at t_0
                        v1_ = interpcc1(vel1_, mb1_, x1_, x2_, x3_)
                        v2_ = interpcc2(vel2_, mb2_, x1_, x2_, x3_)
                        v_ = v1_ + (v2_ - v1_) * (time_ / delta_t_)
                        derv1_, derv2_, derv3_ = vel2derv(x1_, x2_, x3_, v_[0], v_[1], v_[2])

                        # calculate time step
                        lidx11_, lidx21_, lidx31_ = mesh2lidx1(mb1_, x1_, x2_, x3_)
                        lidx12_, lidx22_, lidx32_ = mesh2lidx2(mb2_, x1_, x2_, x3_)
                        dx1f_ = min(dx1f1[mb1_, int(lidx11_)], dx1f2[mb2_, int(lidx12_)])
                        dx2f_ = min(dx2f1[mb1_, int(lidx21_)], dx2f2[mb2_, int(lidx22_)])
                        dx3f_ = min(dx3f1[mb1_, int(lidx31_)], dx3f2[mb2_, int(lidx32_)])
                        dt_ = min(cfl[0] * dx1f_ / (abs(derv1_) + 1e-10),
                                  cfl[1] * dx2f_ / (abs(derv2_) + 1e-10),
                                  cfl[2] * dx3f_ / (abs(derv3_) + 1e-10),
                                  delta_t_ - time_, 2 * dt_)

                        # integrate half time step
                        time__ = time_ + 0.5 * dt_
                        x1__ = x1_ + sign * 0.5 * dt_ * derv1_
                        x2__ = x2_ + sign * 0.5 * dt_ * derv2_
                        x3__ = x3_ + sign * 0.5 * dt_ * derv3_

                        # get velocity and transform to derivatives at t_1/2
                        v1_ = interpcc1(vel1_, mb1_, x1__, x2__, x3__)
                        v2_ = interpcc2(vel2_, mb2_, x1__, x2__, x3__)
                        v_ = v1_ + (v2_ - v1_) * (time__ / delta_t_)
                        derv1_, derv2_, derv3_ = vel2derv(x1__, x2__, x3__, v_[0], v_[1], v_[2])

                        # integrate full time step
                        time_ = time_ + dt_
                        x1_ = x1_ + sign * 0.5 * dt_ * derv1_
                        x2_ = x2_ + sign * 0.5 * dt_ * derv2_
                        x3_ = x3_ + sign * 0.5 * dt_ * derv3_

                        if (x1_ < x1min_ or x1_ > x1max_
                                or x2_ < x2min_ or x2_ > x2max_
                                or x3_ < x3min_ or x3_ > x3max_):
                            break

                    # apply boundary conditions to remaining particles
                    x1_, x2_, x3_ = boundary(x1_, x2_, x3_)

                    # check if it is out-of-bound
                    if (x1_ < x1minrt or x1_ > x1maxrt
                            or x2_ < x2minrt or x2_ > x2maxrt
                            or x3_ < x3minrt or x3_ > x3maxrt):
                        types_[p_] = 2
                    elif time_ < delta_t_:
                        types_[p_] = 1
                    else:
                        types_[p_] = 0

                    meshs_[p_, 0], meshs_[p_, 1], meshs_[p_, 2] = x1_, x2_, x3_
                    time_elapsed_[p_] = time_

                # move particles according to types
                order_ = np.argsort(types_[finished_:in_bound_])
                order_ += finished_
                pids_[finished_:in_bound_] = pids_[order_]
                meshs_[finished_:in_bound_] = meshs_[order_, :]
                time_elapsed_[finished_:in_bound_] = time_elapsed_[order_]
                types_[finished_:in_bound_] = types_[order_]

                finished_ = np.searchsorted(types_[finished_:in_bound_], 1) + finished_
                in_bound_ = np.searchsorted(types_[finished_:in_bound_], 2) + finished_
                remaining_ = in_bound_ - finished_
            return in_bound_
        return compute
