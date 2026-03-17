import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def collect_nearby_indices(x, y, ids, alive, qx, qy, radius_sq, exclude_id, out_idx):
    count = 0
    n = x.shape[0]

    for i in range(n):
        if not alive[i]:
            continue
        if exclude_id >= 0 and ids[i] == exclude_id:
            continue

        dx = x[i] - qx
        dy = y[i] - qy
        if dx * dx + dy * dy <= radius_sq:
            out_idx[count] = i
            count += 1

    return count


@njit(cache=True, fastmath=True)
def nearest_alive_index(x, y, ids, alive, qx, qy, exclude_id):
    best = -1
    best_d2 = 1e30
    n = x.shape[0]

    for i in range(n):
        if not alive[i]:
            continue
        if exclude_id >= 0 and ids[i] == exclude_id:
            continue

        dx = x[i] - qx
        dy = y[i] - qy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = i

    return best


@njit(cache=True, fastmath=True)
def nearest_alive_grown_index(x, y, ids, alive, grown, qx, qy, exclude_id):
    best = -1
    best_d2 = 1e30
    n = x.shape[0]

    for i in range(n):
        if not alive[i]:
            continue
        if not grown[i]:
            continue
        if exclude_id >= 0 and ids[i] == exclude_id:
            continue

        dx = x[i] - qx
        dy = y[i] - qy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = i

    return best
