import numpy as np
from _fastkernels import (
    solve_collisions_same_species_raw as _solve_raw,
    collect_nearby_indices,
    nearest_alive_index,
    nearest_alive_grown_index,
)


def solve_collisions_same_species(x, y, vx, vy, radius, speed, alive, width, height):
    touched = _solve_raw(x, y, vx, vy, radius, speed, alive, width, height)
    return np.frombuffer(touched, dtype=np.uint8).copy()
