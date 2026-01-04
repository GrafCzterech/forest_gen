import numpy as np
from collections import deque

class FlowAccumulator:
    """
    D8 single-flow accumulation model.

    Computes flow accumulation using the D8 algorithm, where each cell
    drains to its single steepest downslope neighbor.
    """

    def compute(self, hm: np.ndarray) -> np.ndarray:
        """
        Compute D8 flow accumulation from a heightmap.

        Each cell contributes a unit flow to itself and all downstream
        cells along the steepest descent path.

        :param hm: Terrain heightmap.
        :type hm: numpy.ndarray
        :return: Flow accumulation array.
        :rtype: numpy.ndarray
        """
        hm = np.asarray(hm)
        rows, cols = hm.shape
        n = rows * cols

        rt2 = float(np.sqrt(2.0))
        nbrs = [
            (-1, -1, rt2),
            (-1,  0, 1.0),
            (-1,  1, rt2),
            ( 0, -1, 1.0),
            ( 0,  1, 1.0),
            ( 1, -1, rt2),
            ( 1,  0, 1.0),
            ( 1,  1, rt2),
        ]

        def lin(i: int, j: int) -> int:
            return i * cols + j

        receiver = np.empty(n, dtype=np.int64)

        for i in range(rows):
            for j in range(cols):
                h0 = float(hm[i, j])
                best_i, best_j = i, j
                best_slope = 0.0 
                for di, dj, dist in nbrs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        drop = h0 - float(hm[ni, nj])
                        if drop > 0.0:
                            s = drop / dist
                            if s > best_slope:
                                best_slope = s
                                best_i, best_j = ni, nj

                receiver[lin(i, j)] = lin(best_i, best_j)

        indeg = np.zeros(n, dtype=np.int32)
        for k in range(n):
            r = receiver[k]
            if r != k:
                indeg[r] += 1

        acc = np.ones(n, dtype=np.float64)
        q = deque(int(k) for k in range(n) if indeg[k] == 0)

        processed = 0
        while q:
            k = q.popleft()
            processed += 1
            r = receiver[k]
            if r != k:
                acc[r] += acc[k]
                indeg[r] -= 1
                if indeg[r] == 0:
                    q.append(r)

        if processed != n:
            raise RuntimeError(
                "Flow accumulation encountered a cycle"
            )

        return acc.reshape((rows, cols))
