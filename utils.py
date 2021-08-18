import numba as nb
import numpy as np
import itertools as it
from frame import Frame
from particles import Particles


def poisson_disk_sampler(frame: Frame, par: Particles, radius=0.8, seed=None):
    """
    Provided the frame and existing particles, generate a blue noise smaple in the unoccupied regions and append to
    the end of the array using (modified) Bridson's algorithm:
    Fast Poisson Disk Sampling in Arbitrary Dimensions (https://dl.acm.org/doi/10.1145/1278780.1278807)

    :param frame: Frame object
    :param par: Particles object containing existing particles
    :param radius: minimum distance between particles
    :param seed: seed for random number generation
    :return: None
    """

    if seed is not None:
        np.random.seed(seed)

    nx_root = frame.header['RootGridSize'].astype(np.int64)
    nx = frame.header['MeshBlockSize'].astype(np.int64)
    nx[nx > 1] -= 2 * frame.num_ghost
    ndim = np.sum(nx > 1)

    nmb = np.int64(frame.header['NumMeshBlocks'])
    maxlevel = np.int64(frame.header['MaxLevel'])
    levels = frame.header['Levels'].astype(np.int64)
    llocs = frame.header['LogicalLocations'].astype(np.int64)

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

    # randomly select an initial sample if none exists
    if len(ptpos) == 0:
        newpos = np.random.rand(3) * nx_root
        newpos[ndim:] = 0.0
        add_point(newpos, pictable, nxt, ptpos)

    # apply Bridson's algorithm
    # randomly select k candidates adjacent to an active point
    # reject if too close to other points or out of bound, otherwise, add to the active list
    @nb.njit
    def has_nearby(gidx_, pictable_, nxt_, ptpos_, radius_):
        mblist_ = nb.typed.List()
        offset_ = np.zeros(3, dtype=np.int64)
        for iofs_ in [-1, 1]:
            for jofs_ in [-1, 1] if ndim >= 2 else [0]:
                for kofs_ in [-1, 1] if ndim >= 3 else [0]:
                    offset_[0], offset_[1], offset_[2] = iofs_, jofs_, kofs_
                    corner_ = gidx_ + offset_ * radius_
                    lloc_ = (corner_ / nx_root * mbtable_shape).astype(np.int64)
                    lloc_[0] = min(max(0, lloc_[0]), mbtable_shape[0] - 1)
                    lloc_[1] = min(max(0, lloc_[1]), mbtable_shape[1] - 1)
                    lloc_[2] = min(max(0, lloc_[2]), mbtable_shape[2] - 1)
                    mb_ = mbtable[lloc_[0], lloc_[1], lloc_[2]]
                    if mb_ not in mblist_:
                        mblist_.append(mb_)
        rsq_ = radius_ * radius_
        left_ = np.copy(gidx_)
        left_[:ndim] -= radius
        right_ = np.copy(gidx_)
        right_[:ndim] += radius
        for mb_ in mblist_:
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

    @nb.njit
    def modified_bridson(pictable_, nxt_, ptpos_, radius_=radius, k=16):
        active_ = 0
        while active_ < len(ptpos_):
            lloc_ = (ptpos_[active_] / nx_root * mbtable_shape).astype(np.int64)
            mb_ = mbtable[lloc_[0], lloc_[1], lloc_[2]]
            rmin_ = np.ldexp(radius_, -levels[mb_])
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
    dn = len(ptpos) - par.size
    par.resize(len(ptpos))

    gidx2mesh = frame.global_indices_to_mesh_position
    par.gidxs[-dn:] = ptpos[-dn:]
    par.meshs[-dn:, 0], par.meshs[-dn:, 1], par.meshs[-dn:, 2] = \
        gidx2mesh(par.gidxs[-dn:, 0], par.gidxs[-dn:, 1], par.gidxs[-dn:, 2])
