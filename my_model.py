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
SHEEP_IDLE_PATH = ASSET_DIR / "sheep.png"
SHEEP_RUN_PATH = ASSET_DIR / "sheep_long.png"

NUM_SHEEP = 5
SHEEP_SCALE = 54  # final rendered width/height in pixels
SHEEP_SPEED = 95.0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


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


def angle_diff_deg(current: float, target: float) -> float:
    """Smallest signed angular difference from current to target."""
    return (target - current + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------
# Sheep agent
# ------------------------------------------------------------
class Sheep:
    def __init__(self, idle_sprite: pygame.Surface, run_sprite: pygame.Surface):
        self.idle_sprite = idle_sprite
        self.run_sprite = run_sprite

        self.pos = Vector2(
            random.uniform(0, WIDTH),
            random.uniform(0, HEIGHT),
        )

        initial_angle = random.uniform(0, math.tau)
        initial_speed = random.uniform(40.0, 90.0)
        self.vel = Vector2(math.cos(initial_angle), math.sin(initial_angle)) * initial_speed
        self.acc = Vector2()

        # Vehicle / steering parameters
        self.max_speed = 120.0
        self.max_force = 85.0
        self.drag = 0.92

        # Wander behavior parameters
        self.wander_angle = random.uniform(0, math.tau)
        self.wander_rate = 1.7
        self.wander_radius = 28.0
        self.wander_distance = 52.0

        # Smooth sprite orientation in degrees.
        # Assumes the animal faces "up" in the sprite image.
        self.display_angle = math.degrees(math.atan2(-self.vel.x, self.vel.y))
        self.max_turn_speed = 320.0  # degrees per second

        # Fixed collision radius based on the round/original sheep body,
        # even when the elongated sprite is shown while moving.
        self.animation_phase = random.uniform(0.0, math.tau)
        self.base_radius = SHEEP_SCALE * 0.38

    def apply_force(self, force: Vector2) -> None:
        self.acc += force

    def wander(self, dt: float) -> None:
        # Continuous steering / vehicle-model style wander.
        if self.vel.length_squared() < 1e-6:
            heading = Vector2(0, -1)
        else:
            heading = self.vel.normalize()

        self.wander_angle += random.uniform(-1.0, 1.0) * self.wander_rate * dt

        circle_center = heading * self.wander_distance
        offset = Vector2(math.cos(self.wander_angle), math.sin(self.wander_angle)) * self.wander_radius
        desired = circle_center + offset

        if desired.length_squared() > 1e-6:
            desired = desired.normalize() * self.max_speed
            steering = desired - self.vel
            if steering.length() > self.max_force:
                steering.scale_to_length(self.max_force)
            self.apply_force(steering)

    def update(self, dt: float) -> None:
        self.wander(dt)

        self.vel += self.acc * dt
        self.acc.update(0.0, 0.0)

        # Soft drag keeps the sheep from running at max speed all the time.
        self.vel *= self.drag

        speed = self.vel.length()
        if speed > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        elif speed < 28.0:
            # Give a gentle push if the sheep slows down too much.
            heading = Vector2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
            if heading.length_squared() > 1e-6:
                heading = heading.normalize()
                self.vel += heading * 20.0 * dt

        self.pos += self.vel * dt
        self.pos, self.vel = bounce_in_bounds(
            self.pos,
            self.vel,
            self.base_radius,
            WIDTH,
            HEIGHT,
        )

        # Smoothly rotate to match movement direction.
        if self.vel.length_squared() > 1e-6:
            target_angle = math.degrees(math.atan2(-self.vel.x, self.vel.y))
            delta = angle_diff_deg(self.display_angle, target_angle)
            max_step = self.max_turn_speed * dt
            self.display_angle += clamp(delta, -max_step, max_step)

        self.animation_phase += speed * 0.06 * dt

    def draw(self, screen: pygame.Surface) -> None:
        speed = self.vel.length()
        speed_ratio = clamp(speed / self.max_speed, 0.0, 1.0)

        # Switch to elongated sprite when moving faster.
        if speed_ratio > 0.42:
            base_sprite = self.run_sprite
        else:
            base_sprite = self.idle_sprite

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


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Wolf-Sheep Prototype")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    sheep_idle = load_sprite(SHEEP_IDLE_PATH, SHEEP_SCALE)
    sheep_run = load_sprite(SHEEP_RUN_PATH, SHEEP_SCALE)

    flock = [Sheep(sheep_idle, sheep_run) for _ in range(NUM_SHEEP)]

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
