import itertools as it

import numba as nb
import numpy as np

from .frame import Frame
from .particles import Particles


def poisson_disk_sampler(frame: Frame, par: Particles, radius=1.0, mindist=None, seed=None, flag=None):
    """
    Provided the frame and existing particles, generate a blue noise smaple in the unoccupied regions and append to
    the end of the array using (modified) Bridson's algorithm:
    Fast Poisson Disk Sampling in Arbitrary Dimensions (https://dl.acm.org/doi/10.1145/1278780.1278807)

    :param frame: Frame object
    :param par: Particles object containing existing particles
    :param radius: minimum distance between particles
    :param mindist: minimum distance as an array, override radius if set
    :param seed: number of initial sample to be put in
    :param flag: flag for newly generated particles
    :return: None
    """

    nx_root = frame.header['RootGridSize'].astype(np.int64)
    nx = frame.header['MeshBlockSize'].astype(np.int64)
    nx[nx > 1] -= 2 * frame.num_ghost
    ndim = np.sum(nx > 1)

    nmb = np.int64(frame.header['NumMeshBlocks'])
    maxlevel = np.int64(frame.header['MaxLevel'])
    levels = frame.header['Levels'].astype(np.int64)
    llocs = frame.header['LogicalLocations'].astype(np.int64)

    if mindist is None:
        mindist = np.empty([nmb] + nx.tolist(), dtype=np.float64)
        mindist[...] = np.ldexp(radius, -levels[:, None, None, None])

    # set up hashtable for locating meshblock
    mbtable_shape = nx_root // nx
    mbtable_shape[nx > 1] <<= maxlevel
    mbtable = np.empty(mbtable_shape, dtype=np.int64)

    for mb in range(nmb):
        left = llocs[mb] << (maxlevel - levels[mb])
        right = (llocs[mb] + 1) << (maxlevel - levels[mb])
        mbtable[tuple(it.starmap(slice, zip(left, right)))] = mb

    # set up hashtable for points-in-cell (pic) linked list and point list
    pictable_shape = tuple((nmb, *nx))
    pictable = np.full(pictable_shape, -1)
    nxt = nb.typed.List().empty_list(nb.int64)
    ptpos = nb.typed.List().empty_list(nb.types.Array(dtype=nb.float64, ndim=1, layout="C"))

    # convert to global index and hash existing points
    mesh2gidx = frame.mesh_position_to_global_indices
    par.gidxs[:, 0], par.gidxs[:, 1], par.gidxs[:, 2] = mesh2gidx(par.meshs[:, 0], par.meshs[:, 1], par.meshs[:, 2])

    @nb.njit
    def add_point(gidx_, pictable_, nxt_, ptpos_):
        lloc_ = (gidx_ / nx_root * mbtable_shape).astype(np.int64)
        mb_ = mbtable[lloc_[0], lloc_[1], lloc_[2]]
        lidx_ = np.ldexp(gidx_ - np.ldexp(lloc_ * nx, -maxlevel), levels[mb_]).astype(np.int64)
        p_ = len(ptpos_)
        head_ = pictable_[mb_, lidx_[0], lidx_[1], lidx_[2]]
        pictable_[mb_, lidx_[0], lidx_[1], lidx_[2]] = p_
        nxt_.append(head_)
        ptpos_.append(np.copy(gidx_))

    @nb.njit
    def hash_points(gidxs_, pictable_, nxt_, ptpos_):
        for p_ in range(gidxs_.shape[0]):
            add_point(gidxs_[p_], pictable_, nxt_, ptpos_)

    hash_points(par.gidxs, pictable, nxt, ptpos)

    @nb.njit
    def has_nearby(gidx_, pictable_, nxt_, ptpos_, radius_):
        left_ = np.copy(gidx_)
        left_[:ndim] -= radius_
        right_ = np.copy(gidx_)
        right_[:ndim] += radius_

        left_lloc_ = (left_ / nx_root * mbtable_shape).astype(np.int64)
        left_lloc_[0] = min(max(0, left_lloc_[0]), mbtable_shape[0] - 1)
        left_lloc_[1] = min(max(0, left_lloc_[1]), mbtable_shape[1] - 1)
        left_lloc_[2] = min(max(0, left_lloc_[2]), mbtable_shape[2] - 1)

        right_lloc_ = (right_ / nx_root * mbtable_shape).astype(np.int64)
        right_lloc_[0] = min(max(0, right_lloc_[0]), mbtable_shape[0] - 1)
        right_lloc_[1] = min(max(0, right_lloc_[1]), mbtable_shape[1] - 1)
        right_lloc_[2] = min(max(0, right_lloc_[2]), mbtable_shape[2] - 1)

        rsq_ = radius_ * radius_
        for mb_ in np.unique(mbtable[left_lloc_[0]:right_lloc_[0] + 1,
                             left_lloc_[1]:right_lloc_[1] + 1,
                             left_lloc_[2]:right_lloc_[2] + 1]):
            lidx_left_ = np.ldexp(left_ - np.ldexp(llocs[mb_] * nx, -maxlevel), levels[mb_]).astype(np.int64)
            lidx_left_[0] = min(max(0, lidx_left_[0]), nx[0] - 1)
            lidx_left_[1] = min(max(0, lidx_left_[1]), nx[1] - 1)
            lidx_left_[2] = min(max(0, lidx_left_[2]), nx[2] - 1)
            lidx_right_ = np.ldexp(right_ - np.ldexp(llocs[mb_] * nx, -maxlevel), levels[mb_]).astype(np.int64)
            lidx_right_[0] = min(max(0, lidx_right_[0]), nx[0] - 1)
            lidx_right_[1] = min(max(0, lidx_right_[1]), nx[1] - 1)
            lidx_right_[2] = min(max(0, lidx_right_[2]), nx[2] - 1)
            for i_ in range(lidx_left_[0], lidx_right_[0] + 1):
                for j_ in range(lidx_left_[1], lidx_right_[1] + 1):
                    for k_ in range(lidx_left_[2], lidx_right_[2] + 1):
                        p_ = pictable_[mb_, i_, j_, k_]
                        while p_ >= 0:
                            if np.sum(np.square(ptpos_[p_] - gidx_)) < rsq_:
                                return True
                            p_ = nxt_[p_]
        return False

    # randomly select initial samples
    if seed is None:
        seed = 1 if len(ptpos) == 0 else 0

    @nb.njit
    def pure_random(pictable_, nxt_, ptpos_):
        if seed == 0:
            return
        cumweight_ = np.cumsum(mindist ** -ndim)
        cumweight_ /= cumweight_[-1]
        indices_ = np.searchsorted(cumweight_, np.random.rand(seed))
        strides = np.cumprod(np.array(mindist.shape[::-1]))
        mbs_ = indices_ // strides[2]
        indices_ -= mbs_ * strides[2]
        ks_ = indices_ // strides[1]
        indices_ -= ks_ * strides[1]
        js_ = indices_ // strides[0]
        indices_ -= js_ * strides[0]
        is_ = indices_
        for mb_, k_, j_, i_ in zip(mbs_, ks_, js_, is_):
            newpos_ = np.ldexp(
                (llocs[mb_] << maxlevel) / mbtable_shape * nx_root + np.array([i_, j_, k_]) + (np.random.rand(3) - 0.5),
                -levels[mb_])
            newpos_[ndim:] = 0.0
            lloc_ = (newpos_ / nx_root * mbtable_shape).astype(np.int64)
            mb_ = mbtable[lloc_[0], lloc_[1], lloc_[2]]
            lidx_ = np.ldexp(newpos_ - np.ldexp(llocs[mb_] * nx, -maxlevel), levels[mb_]).astype(np.int64)
            lidx_[0] = min(max(0, lidx_[0]), nx[0] - 1)
            lidx_[1] = min(max(0, lidx_[1]), nx[1] - 1)
            lidx_[2] = min(max(0, lidx_[2]), nx[2] - 1)
            rmin_ = mindist[mb_, lidx_[2], lidx_[1], lidx_[0]]
            if (np.any(newpos_ < 0) or np.any(newpos_ > nx_root)
                    or has_nearby(newpos_, pictable_, nxt_, ptpos_, rmin_)):
                continue
            add_point(newpos_, pictable_, nxt_, ptpos_)

    pure_random(pictable, nxt, ptpos)

    # apply Bridson's algorithm
    # randomly select k candidates adjacent to an active point
    # reject if too close to other points or out of bound, otherwise, add to the active list
    @nb.njit
    def modified_bridson(pictable_, nxt_, ptpos_, k=16):
        active_ = 0
        while active_ < len(ptpos_):
            lloc_ = (ptpos_[active_] / nx_root * mbtable_shape).astype(np.int64)
            mb_ = mbtable[lloc_[0], lloc_[1], lloc_[2]]
            lidx_ = np.ldexp(ptpos_[active_] - np.ldexp(llocs[mb_] * nx, -maxlevel), levels[mb_]).astype(np.int64)
            lidx_[0] = min(max(0, lidx_[0]), nx[0] - 1)
            lidx_[1] = min(max(0, lidx_[1]), nx[1] - 1)
            lidx_[2] = min(max(0, lidx_[2]), nx[2] - 1)
            rmin_ = mindist[mb_, lidx_[2], lidx_[1], lidx_[0]]
            for i in range(k):
                newpos_ = np.copy(ptpos_[active_])
                if ndim == 2:
                    q1_, q2_ = np.random.rand(2)
                    r_ = rmin_ * (q1_ + 1)
                    phi_ = 2 * np.pi * q2_
                    newpos_[0] += r_ * np.cos(phi_)
                    newpos_[1] += r_ * np.sin(phi_)
                elif ndim == 3:
                    q1_, q2_, q3_ = np.random.rand(3)
                    r_ = rmin_ * (q1_ + 1)
                    costheta = 2 * q2_ - 1
                    sintheta = np.sqrt(1 - costheta * costheta)
                    phi_ = 2 * np.pi * q3_
                    newpos_[0] += r_ * sintheta * np.cos(phi_)
                    newpos_[1] += r_ * sintheta * np.phi(phi_)
                    newpos_[2] += r_ * costheta
                if (np.any(newpos_ < 0) or np.any(newpos_ > nx_root)
                        or has_nearby(newpos_, pictable_, nxt_, ptpos_, rmin_)):
                    continue
                add_point(newpos_, pictable_, nxt_, ptpos_)
            active_ += 1

    modified_bridson(pictable, nxt, ptpos)

    # resize, append and convert back to physical coordinate
    if (len(ptpos) > par.size):
        dn = len(ptpos) - par.size
        par.resize(len(ptpos))

        gidx2mesh = frame.global_indices_to_mesh_position
        par.gidxs[-dn:] = ptpos[-dn:]
        par.meshs[-dn:, 0], par.meshs[-dn:, 1], par.meshs[-dn:, 2] = \
            gidx2mesh(par.gidxs[-dn:, 0], par.gidxs[-dn:, 1], par.gidxs[-dn:, 2])

        par.flags[-dn:] = flag


def _convert_to_serializable(x):
    if isinstance(x, np.bytes_) or isinstance(x, bytes):
        out_ = x.decode('ascii', 'replace')
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


def serialize(x):
    import json
    return json.dumps(_convert_to_serializable(x))
