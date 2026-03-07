import math
import random
from pathlib import Path

import pygame
from pygame.math import Vector2


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (78, 145, 68)

ASSET_DIR = Path("img")
SHEEP_ANIM_DIR = ASSET_DIR / "animation" / "sheep"
SHEEP_ANIM_FRAME_COUNT = 120
SHEEP_ANIM_CYCLE_SEC = 0.5
SHEEP_ANIM_FPS = SHEEP_ANIM_FRAME_COUNT / SHEEP_ANIM_CYCLE_SEC

NUM_SHEEP = 10
SHEEP_SCALE = 72
SHEEP_SPEED = 95.0
TURN_DURATION_SEC = 0.5
STEP_SPEED_MULT_EXPAND = 0.7
STEP_SPEED_MULT_COMPRESS = 2-STEP_SPEED_MULT_EXPAND


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def angle_diff_deg(current: float, target: float) -> float:
    """Smallest signed angular difference from current to target."""
    return (target - current + 180.0) % 360.0 - 180.0


def bounce_in_bounds(pos: Vector2, vel: Vector2, radius: float, width: int, height: int) -> tuple[Vector2, Vector2]:
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


def load_sheep_animation_frames(directory: Path, size: int, frame_count: int) -> list[pygame.Surface]:
    frames: list[pygame.Surface] = []
    for i in range(frame_count):
        frame_path = directory / f"sheep{i:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing sheep animation frame: {frame_path}")
        frames.append(load_sprite(frame_path, size))
    return frames


# ------------------------------------------------------------
# Sheep agent
# ------------------------------------------------------------
class Sheep:
    def __init__(self, animation_frames: list[pygame.Surface]):
        self.animation_frames = animation_frames

        self.pos = Vector2(
            random.uniform(0, WIDTH),
            random.uniform(0, HEIGHT),
        )

        # Random initial direction, fixed speed for all sheep.
        initial_angle = random.uniform(0, math.tau)
        self.speed = SHEEP_SPEED
        self.vel = Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed

        # Sprite orientation in degrees.
        # Assumes the animal faces "up" in the sprite image.
        self.display_angle = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        self.target_angle = self.display_angle
        self.turn_start_angle = self.display_angle
        self.turn_elapsed = TURN_DURATION_SEC

        # Start each sheep at a random frame offset so they do not look synchronized.
        self.anim_frame_cursor = random.uniform(0.0, float(len(self.animation_frames)))

        # Fixed collision radius based on the round/original sheep body.
        self.base_radius = SHEEP_SCALE * 0.38

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
            return STEP_SPEED_MULT_EXPAND
        return STEP_SPEED_MULT_COMPRESS

    def update(self, dt: float) -> None:
        # Keep total direction speed constant, then scale displacement by gait phase.
        if self.vel.length_squared() > 1e-6:
            self.vel = self.vel.normalize() * self.speed
            step_speed_mult = self.step_speed_multiplier()
        else:
            step_speed_mult = 0.0

        self.pos += self.vel * dt * step_speed_mult
        self.pos, self.vel = bounce_in_bounds(
            self.pos,
            self.vel,
            self.base_radius,
            WIDTH,
            HEIGHT,
        )

        self.retarget_orientation()
        self.update_orientation(dt)

        # Advance animation at authored rate while moving.
        if self.vel.length_squared() > 1e-6:
            self.anim_frame_cursor = (self.anim_frame_cursor + SHEEP_ANIM_FPS * dt) % len(self.animation_frames)

    def draw(self, screen: pygame.Surface) -> None:
        frame_index = int(self.anim_frame_cursor) % len(self.animation_frames)
        base_sprite = self.animation_frames[frame_index]

        render_angle = -self.display_angle
        rotated = pygame.transform.rotozoom(base_sprite, render_angle, 1.0)
        rect = rotated.get_rect(center=(self.pos.x, self.pos.y))

        screen.blit(rotated, rect)

        # Direction / heading line follows the exact rendered sprite angle.
        # The sprite art faces downward in its source image, so angle 0 means +Y.
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

        # Optional subtle debug ring for body radius / collision proxy.
        # Uncomment during tuning:
        # pygame.draw.circle(screen, (220, 255, 220), self.pos, self.base_radius, 1)


def resolve_sheep_collisions(flock: list[Sheep]) -> None:
    for i in range(len(flock)):
        a = flock[i]
        for j in range(i + 1, len(flock)):
            b = flock[j]

            delta = b.pos - a.pos
            min_dist = a.base_radius + b.base_radius
            dist_sq = delta.length_squared()

            if dist_sq >= min_dist * min_dist:
                continue

            if dist_sq < 1e-12:
                # Rare exact overlap fallback.
                angle = random.uniform(0.0, math.tau)
                normal = Vector2(math.cos(angle), math.sin(angle))
                dist = 0.0
            else:
                dist = math.sqrt(dist_sq)
                normal = delta / dist

            # Positional correction to avoid sticky overlap.
            overlap = min_dist - dist
            if overlap > 0.0:
                correction = normal * (overlap * 0.5)
                a.pos -= correction
                b.pos += correction

                a.pos, a.vel = bounce_in_bounds(a.pos, a.vel, a.base_radius, WIDTH, HEIGHT)
                b.pos, b.vel = bounce_in_bounds(b.pos, b.vel, b.base_radius, WIDTH, HEIGHT)

            # Elastic equal-mass collision: swap normal velocity components.
            va_n = a.vel.dot(normal)
            vb_n = b.vel.dot(normal)

            # Apply only if moving toward each other along collision normal.
            if va_n - vb_n > 0.0:
                a.vel += (vb_n - va_n) * normal
                b.vel += (va_n - vb_n) * normal

                # Keep every sheep at constant speed.
                if a.vel.length_squared() > 1e-6:
                    a.vel = a.vel.normalize() * a.speed
                    a.retarget_orientation()

                if b.vel.length_squared() > 1e-6:
                    b.vel = b.vel.normalize() * b.speed
                    b.retarget_orientation()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Wolf-Sheep Prototype")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    sheep_animation_frames = load_sheep_animation_frames(
        SHEEP_ANIM_DIR,
        SHEEP_SCALE,
        SHEEP_ANIM_FRAME_COUNT,
    )

    flock = [Sheep(sheep_animation_frames) for _ in range(NUM_SHEEP)]

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        for sheep in flock:
            sheep.update(dt)

        resolve_sheep_collisions(flock)

        screen.fill(BG_COLOR)

        for sheep in flock:
            sheep.draw(screen)

        fps_text = font.render(f"FPS: {clock.get_fps():5.1f}", True, (240, 245, 240))
        sheep_text = font.render(f"Sheep: {len(flock)}", True, (240, 245, 240))
        screen.blit(fps_text, (12, 10))
        screen.blit(sheep_text, (12, 32))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
