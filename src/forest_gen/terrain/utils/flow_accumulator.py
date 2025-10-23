import numpy as np


class FlowAccumulator:
    """D8 single-flow accumulation."""

    def compute(self, hm: np.ndarray) -> np.ndarray:
        rows, cols = hm.shape
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        flow_to = np.zeros((rows, cols, 2), dtype=int)
        sink = np.zeros((rows, cols), dtype=bool)

        for i in range(rows):
            for j in range(cols):
                mh = hm[i, j]
                best = (0, 0)
                for di, dj in dirs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and hm[ni, nj] < mh:
                        mh, best = hm[ni, nj], (di, dj)
                flow_to[i, j] = best
                sink[i, j] = best == (0, 0)

        acc = np.ones((rows, cols), dtype=float)
        visited = {}

        def dfs(i, j):
            if (i, j) in visited:
                return visited[(i, j)]
            if sink[i, j]:
                val = acc[i, j]
            else:
                di, dj = flow_to[i, j]
                val = acc[i, j] + dfs(i + di, j + dj)
            visited[(i, j)] = val
            return val

        for i in range(rows):
            for j in range(cols):
                dfs(i, j)

        for (i, j), v in visited.items():
            acc[i, j] = v

        return acc
