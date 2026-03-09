import math
import random
from typing import Protocol

from pygame.math import Vector2


# ------------------------------------------------------------
# Movement Motors
# ------------------------------------------------------------
class MovementMotor(Protocol):
    def advance(
        self,
        pos: Vector2,
        vel: Vector2,
        speed: float,
        dt: float,
        radius: float,
        displacement_scale: float,
    ) -> tuple[Vector2, Vector2]: ...


class StraightLineMotor:
    def advance(
        self,
        pos: Vector2,
        vel: Vector2,
        speed: float,
        dt: float,
        radius: float,
        displacement_scale: float,
    ) -> tuple[Vector2, Vector2]:
        if vel.length_squared() < 1e-6:
            angle = random.uniform(0.0, math.tau)
            direction = Vector2(math.cos(angle), math.sin(angle))
            new_vel = direction * speed
        else:
            new_vel = vel.normalize() * speed

        new_pos = pos + new_vel * dt * displacement_scale
        return new_pos, new_vel


class RandomWalkMotor:
    def __init__(self, turn_rate_rad_per_sec: float = 30.0):
        self.turn_rate = turn_rate_rad_per_sec

    def advance(
        self,
        pos: Vector2,
        vel: Vector2,
        speed: float,
        dt: float,
        radius: float,
        displacement_scale: float,
    ) -> tuple[Vector2, Vector2]:
        if vel.length_squared() < 1e-6:
            angle = random.uniform(0.0, math.tau)
            heading = Vector2(math.cos(angle), math.sin(angle))
        else:
            heading = vel.normalize()

        jitter = random.uniform(-1.0, 1.0) * self.turn_rate * dt
        cos_j = math.cos(jitter)
        sin_j = math.sin(jitter)
        heading = Vector2(
            heading.x * cos_j - heading.y * sin_j,
            heading.x * sin_j + heading.y * cos_j,
        )

        new_vel = heading * speed
        new_pos = pos + new_vel * dt * displacement_scale
        return new_pos, new_vel
