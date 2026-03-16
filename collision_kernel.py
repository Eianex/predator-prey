import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _clamp_and_bounce(px, py, vx, vy, radius, width, height):
    if px < radius:
        px = radius
        vx = -vx
    elif px > width - radius:
        px = width - radius
        vx = -vx

    if py < radius:
        py = radius
        vy = -vy
    elif py > height - radius:
        py = height - radius
        vy = -vy

    return px, py, vx, vy


@njit(cache=True, fastmath=True)
def solve_collisions_same_species(x, y, vx, vy, radius, speed, alive, width, height):
    n = x.shape[0]
    touched = np.zeros(n, dtype=np.uint8)

    for i in range(n - 1):
        if not alive[i]:
            continue

        xi = x[i]
        yi = y[i]
        vxi = vx[i]
        vyi = vy[i]
        ri = radius[i]
        si = speed[i]

        for j in range(i + 1, n):
            if not alive[j]:
                continue

            xj = x[j]
            yj = y[j]
            vxj = vx[j]
            vyj = vy[j]

            dx = xj - xi
            dy = yj - yi
            min_dist = ri + radius[j]
            dist_sq = dx * dx + dy * dy

            if dist_sq >= min_dist * min_dist:
                continue

            if dist_sq < 1e-12:
                nx = 1.0
                ny = 0.0
                dist = 0.0
            else:
                dist = math.sqrt(dist_sq)
                inv = 1.0 / dist
                nx = dx * inv
                ny = dy * inv

            overlap = min_dist - dist
            if overlap > 0.0:
                half = 0.5 * overlap
                corrx = nx * half
                corry = ny * half
                xi -= corrx
                yi -= corry
                xj += corrx
                yj += corry

            xi, yi, vxi, vyi = _clamp_and_bounce(xi, yi, vxi, vyi, ri, width, height)
            xj, yj, vxj, vyj = _clamp_and_bounce(
                xj, yj, vxj, vyj, radius[j], width, height
            )

            va_n = vxi * nx + vyi * ny
            vb_n = vxj * nx + vyj * ny

            if va_n - vb_n > 0.0:
                delta_a = vb_n - va_n
                delta_b = va_n - vb_n

                vxi += delta_a * nx
                vyi += delta_a * ny
                vxj += delta_b * nx
                vyj += delta_b * ny

                la2 = vxi * vxi + vyi * vyi
                if la2 > 1e-12:
                    scale_a = si / math.sqrt(la2)
                    vxi *= scale_a
                    vyi *= scale_a

                sj = speed[j]
                lb2 = vxj * vxj + vyj * vyj
                if lb2 > 1e-12:
                    scale_b = sj / math.sqrt(lb2)
                    vxj *= scale_b
                    vyj *= scale_b

            x[j] = xj
            y[j] = yj
            vx[j] = vxj
            vy[j] = vyj

            touched[i] = 1
            touched[j] = 1

        x[i] = xi
        y[i] = yi
        vx[i] = vxi
        vy[i] = vyi

    return touched
