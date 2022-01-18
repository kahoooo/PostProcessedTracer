import numpy as np


class Particles:
    size: int
    next_id: int
    pids: np.ndarray
    meshs: np.ndarray
    carts: np.ndarray
    gidxs: np.ndarray
    flags: np.ndarray

    def __init__(self, size):
        self.size = size
        self.next_id = size
        self.pids = np.arange(0, size, dtype=np.int64)
        self.meshs = np.empty((size, 3), dtype=np.float64)
        self.carts = np.empty((size, 3), dtype=np.float64)
        self.gidxs = np.empty((size, 3), dtype=np.float64)
        self.flags = np.arange(0, size, dtype='|S1')

    def resize(self, size):
        dsize = size - self.size
        self.size = size
        self.pids.resize(size)
        self.meshs.resize((size, 3))
        self.carts.resize((size, 3))
        self.gidxs.resize((size, 3))
        self.flags.resize(size)
        if dsize > 0:
            self.pids[-dsize:] = np.arange(self.next_id, self.next_id + dsize)
            self.flags[-dsize:] = None
            self.next_id += dsize
