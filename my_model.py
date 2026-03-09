import math
import random
from pathlib import Path
from typing import Protocol

import pygame
from pygame.math import Vector2


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (78, 145, 68)

ASSET_DIR = Path("img")
ANIM_DIR = ASSET_DIR / "animation"
SHEEP_ANIM_DIR = ANIM_DIR / "sheep"
WOLF_ANIM_DIR = ANIM_DIR / "wolf"

ANIM_FRAME_COUNT = 120
ANIM_CYCLE_SEC = 0.5
ANIM_FPS = ANIM_FRAME_COUNT / ANIM_CYCLE_SEC

NUM_SHEEP = 25
NUM_WOLVES = 3
MAX_SHEEP = 160

SHEEP_SCALE = 64
WOLF_SCALE = 64

SHEEP_SPEED = 100.0
WOLF_SPEED = 200.0

TURN_DURATION_SEC = 0.5
SHEEP_STEP_SPEED_MULT_EXPAND = 0.8
SHEEP_STEP_SPEED_MULT_COMPRESS = 2 - SHEEP_STEP_SPEED_MULT_EXPAND
WOLF_STEP_SPEED_MULT_EXPAND = 0.8
WOLF_STEP_SPEED_MULT_COMPRESS = 2 - WOLF_STEP_SPEED_MULT_EXPAND
SHEEP_REPRODUCTION_COOLDOWN_SEC = 5.0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def angle_diff_deg(current: float, target: float) -> float:
    """Smallest signed angular difference from current to target."""
    return (target - current + 180.0) % 360.0 - 180.0


def bounce_in_bounds(
    pos: Vector2, vel: Vector2, radius: float, width: int, height: int
) -> tuple[Vector2, Vector2]:
    if pos.x < radius:
        pos.x = radius
        vel.x *= -1
    elif pos.x > width - radius:
        pos.x = width - radius
        vel.x *= -1

    if pos.y < radius:
        pos.y = radius
        vel.y *= -1
    elif pos.y > height - radius:
        pos.y = height - radius
        vel.y *= -1

    return pos, vel


def load_sprite(path: Path, size: int) -> pygame.Surface:
    surface = pygame.image.load(path.as_posix()).convert_alpha()
    return pygame.transform.smoothscale(surface, (size, size))


def load_animation_frames(
    directory: Path, prefix: str, size: int, frame_count: int
) -> list[pygame.Surface]:
    frames: list[pygame.Surface] = []
    for i in range(frame_count):
        frame_path = directory / f"{prefix}{i:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing animation frame: {frame_path}")
        frames.append(load_sprite(frame_path, size))
    return frames


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
        return bounce_in_bounds(new_pos, new_vel, radius, WIDTH, HEIGHT)


class RandomWalkMotor:
    def __init__(self, turn_rate_rad_per_sec: float = 52.4):
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
        return bounce_in_bounds(new_pos, new_vel, radius, WIDTH, HEIGHT)


# ------------------------------------------------------------
# Collision Contracts
# ------------------------------------------------------------
class CollidableAnimal(Protocol):
    pos: Vector2
    vel: Vector2
    base_radius: float
    speed: float

    def retarget_orientation(self) -> None: ...


def elastic_collision_response(
    a: CollidableAnimal,
    b: CollidableAnimal,
    normal: Vector2,
    dist: float,
    min_dist: float,
) -> None:
    overlap = min_dist - dist
    if overlap > 0.0:
        correction = normal * (overlap * 0.5)
        a.pos -= correction
        b.pos += correction

        a.pos, a.vel = bounce_in_bounds(a.pos, a.vel, a.base_radius, WIDTH, HEIGHT)
        b.pos, b.vel = bounce_in_bounds(b.pos, b.vel, b.base_radius, WIDTH, HEIGHT)

    va_n = a.vel.dot(normal)
    vb_n = b.vel.dot(normal)

    if va_n - vb_n > 0.0:
        a.vel += (vb_n - va_n) * normal
        b.vel += (va_n - vb_n) * normal

        if a.vel.length_squared() > 1e-6:
            a.vel = a.vel.normalize() * a.speed
            a.retarget_orientation()

        if b.vel.length_squared() > 1e-6:
            b.vel = b.vel.normalize() * b.speed
            b.retarget_orientation()


# ------------------------------------------------------------
# Sheep
# ------------------------------------------------------------
class Sheep:
    def __init__(
        self,
        animation_frames: list[pygame.Surface],
        motor: MovementMotor | None = None,
        speed: float = SHEEP_SPEED,
        scale: int = SHEEP_SCALE,
        step_expand: float = SHEEP_STEP_SPEED_MULT_EXPAND,
        step_compress: float = SHEEP_STEP_SPEED_MULT_COMPRESS,
        initial_reproduction_cooldown: float = 0.0,
    ):
        self.animation_frames = animation_frames
        self.motor: MovementMotor = motor if motor is not None else StraightLineMotor()

        self.speed = speed
        self.step_expand = step_expand
        self.step_compress = step_compress
        self.reproduction_cooldown = max(0.0, initial_reproduction_cooldown)

        self.pos = Vector2(
            random.uniform(0, WIDTH),
            random.uniform(0, HEIGHT),
        )

        initial_angle = random.uniform(0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )

        self.display_angle = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        self.target_angle = self.display_angle
        self.turn_start_angle = self.display_angle
        self.turn_elapsed = TURN_DURATION_SEC

        self.anim_frame_cursor = random.uniform(0.0, float(len(self.animation_frames)))
        self.base_radius = scale * 0.38

    def can_reproduce(self) -> bool:
        return self.reproduction_cooldown <= 0.0

    def retarget_orientation(self) -> None:
        if self.vel.length_squared() < 1e-6:
            return

        new_target = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        if abs(angle_diff_deg(self.target_angle, new_target)) > 1e-3:
            self.turn_start_angle = self.display_angle
            self.target_angle = new_target
            self.turn_elapsed = 0.0

    def update_orientation(self, dt: float) -> None:
        if TURN_DURATION_SEC <= 1e-6:
            self.display_angle = self.target_angle
            self.turn_elapsed = TURN_DURATION_SEC
            return

        if self.turn_elapsed >= TURN_DURATION_SEC:
            self.display_angle = self.target_angle
            return

        self.turn_elapsed = min(TURN_DURATION_SEC, self.turn_elapsed + dt)
        progress = self.turn_elapsed / TURN_DURATION_SEC
        delta = angle_diff_deg(self.turn_start_angle, self.target_angle)
        self.display_angle = self.turn_start_angle + delta * progress

    def step_speed_multiplier(self) -> float:
        frame_count = len(self.animation_frames)
        if frame_count <= 0:
            return 1.0

        cycle_phase = (self.anim_frame_cursor % frame_count) / frame_count
        if cycle_phase < 0.5:
            return self.step_expand
        return self.step_compress

    def update(self, dt: float) -> None:
        displacement_scale = (
            self.step_speed_multiplier() if self.vel.length_squared() > 1e-6 else 0.0
        )
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

        self.retarget_orientation()
        self.update_orientation(dt)

        if self.vel.length_squared() > 1e-6:
            self.anim_frame_cursor = (self.anim_frame_cursor + ANIM_FPS * dt) % len(
                self.animation_frames
            )

        if self.reproduction_cooldown > 0.0:
            self.reproduction_cooldown = max(0.0, self.reproduction_cooldown - dt)

    def draw(self, screen: pygame.Surface) -> None:
        frame_index = int(self.anim_frame_cursor) % len(self.animation_frames)
        base_sprite = self.animation_frames[frame_index]

        render_angle = -self.display_angle
        rotated = pygame.transform.rotozoom(base_sprite, render_angle, 1.0)
        rect = rotated.get_rect(center=(self.pos.x, self.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = self.pos
        line_end = self.pos + direction * (self.base_radius + 18)
        pygame.draw.line(
            screen,
            (245, 245, 245),
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            2,
        )


# ------------------------------------------------------------
# Wolf
# ------------------------------------------------------------
class Wolf:
    def __init__(
        self,
        animation_frames: list[pygame.Surface],
        motor: MovementMotor | None = None,
        speed: float = WOLF_SPEED,
        scale: int = WOLF_SCALE,
        step_expand: float = WOLF_STEP_SPEED_MULT_EXPAND,
        step_compress: float = WOLF_STEP_SPEED_MULT_COMPRESS,
    ):
        self.animation_frames = animation_frames
        self.motor: MovementMotor = motor if motor is not None else StraightLineMotor()

        self.speed = speed
        self.step_expand = step_expand
        self.step_compress = step_compress

        self.pos = Vector2(
            random.uniform(0, WIDTH),
            random.uniform(0, HEIGHT),
        )

        initial_angle = random.uniform(0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )

        self.display_angle = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        self.target_angle = self.display_angle
        self.turn_start_angle = self.display_angle
        self.turn_elapsed = TURN_DURATION_SEC

        self.anim_frame_cursor = random.uniform(0.0, float(len(self.animation_frames)))
        self.base_radius = scale * 0.38

    def retarget_orientation(self) -> None:
        if self.vel.length_squared() < 1e-6:
            return

        new_target = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        if abs(angle_diff_deg(self.target_angle, new_target)) > 1e-3:
            self.turn_start_angle = self.display_angle
            self.target_angle = new_target
            self.turn_elapsed = 0.0

    def update_orientation(self, dt: float) -> None:
        if TURN_DURATION_SEC <= 1e-6:
            self.display_angle = self.target_angle
            self.turn_elapsed = TURN_DURATION_SEC
            return

        if self.turn_elapsed >= TURN_DURATION_SEC:
            self.display_angle = self.target_angle
            return

        self.turn_elapsed = min(TURN_DURATION_SEC, self.turn_elapsed + dt)
        progress = self.turn_elapsed / TURN_DURATION_SEC
        delta = angle_diff_deg(self.turn_start_angle, self.target_angle)
        self.display_angle = self.turn_start_angle + delta * progress

    def step_speed_multiplier(self) -> float:
        frame_count = len(self.animation_frames)
        if frame_count <= 0:
            return 1.0

        cycle_phase = (self.anim_frame_cursor % frame_count) / frame_count
        if cycle_phase < 0.5:
            return self.step_expand
        return self.step_compress

    def update(self, dt: float) -> None:
        displacement_scale = (
            self.step_speed_multiplier() if self.vel.length_squared() > 1e-6 else 0.0
        )
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

        self.retarget_orientation()
        self.update_orientation(dt)

        if self.vel.length_squared() > 1e-6:
            self.anim_frame_cursor = (self.anim_frame_cursor + ANIM_FPS * dt) % len(
                self.animation_frames
            )

    def draw(self, screen: pygame.Surface) -> None:
        frame_index = int(self.anim_frame_cursor) % len(self.animation_frames)
        base_sprite = self.animation_frames[frame_index]

        render_angle = -self.display_angle
        rotated = pygame.transform.rotozoom(base_sprite, render_angle, 1.0)
        rect = rotated.get_rect(center=(self.pos.x, self.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = self.pos
        line_end = self.pos + direction * (self.base_radius + 18)
        pygame.draw.line(
            screen,
            (245, 245, 245),
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            2,
        )


# ------------------------------------------------------------
# Interactions
# ------------------------------------------------------------
def spawn_sheep_near(
    par_a: Sheep, par_b: Sheep, sheep_animation_frames: list[pygame.Surface]
) -> Sheep:
    child = Sheep(
        sheep_animation_frames,
        initial_reproduction_cooldown=SHEEP_REPRODUCTION_COOLDOWN_SEC,
    )

    midpoint = (par_a.pos + par_b.pos) * 0.5
    offset = Vector2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
    if offset.length_squared() < 1e-6:
        offset = Vector2(1.0, 0.0)
    offset = offset.normalize() * random.uniform(0.0, par_a.base_radius)

    child.pos = midpoint + offset
    child.pos, child.vel = bounce_in_bounds(
        child.pos, child.vel, child.base_radius, WIDTH, HEIGHT
    )
    return child


def resolve_interactions(
    sheep_flock: list[Sheep],
    wolf_pack: list[Wolf],
    sheep_animation_frames: list[pygame.Surface],
) -> None:
    all_animals: list[Sheep | Wolf] = [*sheep_flock, *wolf_pack]
    sheep_to_remove: set[Sheep] = set()
    newborn_sheep: list[Sheep] = []

    for i in range(len(all_animals)):
        a = all_animals[i]
        if isinstance(a, Sheep) and a in sheep_to_remove:
            continue

        for j in range(i + 1, len(all_animals)):
            b = all_animals[j]
            if isinstance(b, Sheep) and b in sheep_to_remove:
                continue

            delta = b.pos - a.pos
            min_dist = a.base_radius + b.base_radius
            dist_sq = delta.length_squared()

            if dist_sq >= min_dist * min_dist:
                continue

            if isinstance(a, Sheep) and isinstance(b, Wolf):
                sheep_to_remove.add(a)
                continue

            if isinstance(a, Wolf) and isinstance(b, Sheep):
                sheep_to_remove.add(b)
                continue

            if dist_sq < 1e-12:
                angle = random.uniform(0.0, math.tau)
                normal = Vector2(math.cos(angle), math.sin(angle))
                dist = 0.0
            else:
                dist = math.sqrt(dist_sq)
                normal = delta / dist

            if (
                isinstance(a, Sheep)
                and isinstance(b, Sheep)
                and a.can_reproduce()
                and b.can_reproduce()
            ):
                newborn_sheep.append(spawn_sheep_near(a, b, sheep_animation_frames))
                a.reproduction_cooldown = SHEEP_REPRODUCTION_COOLDOWN_SEC
                b.reproduction_cooldown = SHEEP_REPRODUCTION_COOLDOWN_SEC

            elastic_collision_response(a, b, normal, dist, min_dist)

    if sheep_to_remove:
        sheep_flock[:] = [s for s in sheep_flock if s not in sheep_to_remove]

    if newborn_sheep and len(sheep_flock) < MAX_SHEEP:
        slots_left = MAX_SHEEP - len(sheep_flock)
        sheep_flock.extend(newborn_sheep[:slots_left])


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Wolf-Sheep Prototype")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    sheep_animation_frames = load_animation_frames(
        SHEEP_ANIM_DIR,
        "sheep",
        SHEEP_SCALE,
        ANIM_FRAME_COUNT,
    )
    wolf_animation_frames = load_animation_frames(
        WOLF_ANIM_DIR,
        "wolf",
        WOLF_SCALE,
        ANIM_FRAME_COUNT,
    )

    sheep_flock = [
        Sheep(sheep_animation_frames, motor=RandomWalkMotor()) for _ in range(NUM_SHEEP)
    ]
    wolf_pack = [
        Wolf(wolf_animation_frames, motor=StraightLineMotor())
        for _ in range(NUM_WOLVES)
    ]

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        all_animals: list[Sheep | Wolf] = [*sheep_flock, *wolf_pack]

        for animal in all_animals:
            animal.update(dt)

        resolve_interactions(sheep_flock, wolf_pack, sheep_animation_frames)

        all_animals = [*sheep_flock, *wolf_pack]

        screen.fill(BG_COLOR)

        for animal in all_animals:
            animal.draw(screen)

        fps_text = font.render(f"FPS: {clock.get_fps():5.1f}", True, (240, 245, 240))
        sheep_text = font.render(f"Sheep: {len(sheep_flock)}", True, (240, 245, 240))
        wolf_text = font.render(f"Wolves: {len(wolf_pack)}", True, (240, 245, 240))
        screen.blit(fps_text, (12, 10))
        screen.blit(sheep_text, (12, 32))
        screen.blit(wolf_text, (12, 54))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
