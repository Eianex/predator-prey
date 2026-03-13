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
    def __init__(self, turn_interval_sec: float = 3.0):
        self.turn_interval_sec = max(1e-6, turn_interval_sec)
        self.time_to_next_turn = self.turn_interval_sec

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

        self.time_to_next_turn -= dt
        if self.time_to_next_turn <= 0.0:
            current_angle = math.atan2(heading.y, heading.x)
            delta_angle = random.uniform(-math.pi / 4.0, math.pi / 4.0)
            new_angle = current_angle + delta_angle
            heading = Vector2(math.cos(new_angle), math.sin(new_angle))
            while self.time_to_next_turn <= 0.0:
                self.time_to_next_turn += self.turn_interval_sec

        new_vel = heading * speed
        new_pos = pos + new_vel * dt * displacement_scale
        return new_pos, new_vel


class TargetStraightMotor:
    def __init__(self):
        self.target_pos: Vector2 | None = None
        self.target_acquired = False

    def set_target(self, target_pos: Vector2) -> None:
        self.target_pos = Vector2(target_pos.x, target_pos.y)
        self.target_acquired = True

    def clear_target(self) -> None:
        self.target_pos = None
        self.target_acquired = False

    def advance(
        self,
        pos: Vector2,
        vel: Vector2,
        speed: float,
        dt: float,
        radius: float,
        displacement_scale: float,
    ) -> tuple[Vector2, Vector2]:
        if self.target_acquired and self.target_pos is not None:
            to_target = self.target_pos - pos
            dist_sq = to_target.length_squared()
            if dist_sq <= 1e-12:
                self.clear_target()
                new_vel = vel if vel.length_squared() > 1e-6 else Vector2(speed, 0.0)
                return pos, new_vel

            dist = math.sqrt(dist_sq)
            heading = to_target / dist
            new_vel = heading * speed
            step_dist = speed * dt * displacement_scale
            if step_dist >= dist:
                new_pos = Vector2(self.target_pos.x, self.target_pos.y)
                self.clear_target()
            else:
                new_pos = pos + heading * step_dist
            return new_pos, new_vel

        if vel.length_squared() < 1e-6:
            angle = random.uniform(0.0, math.tau)
            heading = Vector2(math.cos(angle), math.sin(angle))
            new_vel = heading * speed
        else:
            new_vel = vel.normalize() * speed

        new_pos = pos + new_vel * dt * displacement_scale
        return new_pos, new_vel
