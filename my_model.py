import math
import random
from pathlib import Path
from typing import Protocol

import pygame
from pygame.math import Vector2


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIDTH, HEIGHT = 1000, 800
FPS = 60
BG_COLOR = (78, 145, 68)

ASSET_DIR = Path("img")
ANIM_DIR = ASSET_DIR / "animation"
SHEEP_ANIM_DIR = ANIM_DIR / "sheep"
WOLF_ANIM_DIR = ANIM_DIR / "wolf"

ANIM_FRAME_COUNT = 120
ANIM_CYCLE_SEC = 0.5
ANIM_FPS = ANIM_FRAME_COUNT / ANIM_CYCLE_SEC

NUM_SHEEP = 50
NUM_WOLVES = 2
MAX_SHEEP = 160

SHEEP_SCALE = 50
WOLF_SCALE = 70

SHEEP_SPEED = 100.0
WOLF_SPEED = 200.0

TURN_DURATION_SEC = 0.5
SHEEP_STEP_SPEED_MULT_EXPAND = 0.8
SHEEP_STEP_SPEED_MULT_COMPRESS = 2 - SHEEP_STEP_SPEED_MULT_EXPAND
WOLF_STEP_SPEED_MULT_EXPAND = 0.8
WOLF_STEP_SPEED_MULT_COMPRESS = 2 - WOLF_STEP_SPEED_MULT_EXPAND
SHEEP_REPRODUCTION_COOLDOWN_SEC = 5.0

# Left metrics panel
PANEL_WIDTH = 320
PANEL_BG_COLOR = (22, 28, 30)
PANEL_BORDER_COLOR = (70, 86, 90)
GRAPH_BG_COLOR = (16, 21, 24)
SHEEP_GRAPH_COLOR = (188, 246, 166)
WOLF_GRAPH_COLOR = (246, 148, 120)
GRAPH_TIME_WINDOW_SEC = 60.0
GRAPH_SAMPLE_INTERVAL_SEC = 0.12


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


def load_image(path: Path, size: int) -> pygame.Surface:
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
        frames.append(load_image(frame_path, size))
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
        self.motor: MovementMotor = motor if motor is not None else RandomWalkMotor()

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
        self.pos, self.vel = bounce_in_bounds(
            self.pos, self.vel, self.base_radius, WIDTH, HEIGHT
        )

        self.retarget_orientation()
        self.update_orientation(dt)

        if self.vel.length_squared() > 1e-6:
            self.anim_frame_cursor = (self.anim_frame_cursor + ANIM_FPS * dt) % len(
                self.animation_frames
            )

        if self.reproduction_cooldown > 0.0:
            self.reproduction_cooldown = max(0.0, self.reproduction_cooldown - dt)

    def draw(self, screen: pygame.Surface, x_offset: int = 0) -> None:
        frame_index = int(self.anim_frame_cursor) % len(self.animation_frames)
        base_image = self.animation_frames[frame_index]

        render_angle = -self.display_angle
        rotated = pygame.transform.rotozoom(base_image, render_angle, 1.0)
        rect = rotated.get_rect(center=(self.pos.x + x_offset, self.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = Vector2(self.pos.x + x_offset, self.pos.y)
        line_end = line_start + direction * (self.base_radius + 18)
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
        self.pos, self.vel = bounce_in_bounds(
            self.pos, self.vel, self.base_radius, WIDTH, HEIGHT
        )

        self.retarget_orientation()
        self.update_orientation(dt)

        if self.vel.length_squared() > 1e-6:
            self.anim_frame_cursor = (self.anim_frame_cursor + ANIM_FPS * dt) % len(
                self.animation_frames
            )

    def draw(self, screen: pygame.Surface, x_offset: int = 0) -> None:
        frame_index = int(self.anim_frame_cursor) % len(self.animation_frames)
        base_image = self.animation_frames[frame_index]

        render_angle = -self.display_angle
        rotated = pygame.transform.rotozoom(base_image, render_angle, 1.0)
        rect = rotated.get_rect(center=(self.pos.x + x_offset, self.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = Vector2(self.pos.x + x_offset, self.pos.y)
        line_end = line_start + direction * (self.base_radius + 18)
        pygame.draw.line(
            screen,
            (245, 245, 245),
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            2,
        )


# ------------------------------------------------------------
# Population Graph Panel
# ------------------------------------------------------------
class PopulationGraph:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        color: tuple[int, int, int],
        initial_value: int,
    ):
        self.rect = rect
        self.label = label
        self.color = color
        self.samples: list[tuple[float, float]] = [(0.0, float(initial_value))]
        self.display_latest = float(initial_value)
        self.display_y_max = max(10.0, float(initial_value) + 3.0)

    def add_sample(self, time_sec: float, value: int) -> None:
        self.samples.append((time_sec, float(value)))
        cutoff = time_sec - GRAPH_TIME_WINDOW_SEC - 1.0
        while len(self.samples) > 2 and self.samples[1][0] < cutoff:
            self.samples.pop(0)

    def update(self, dt: float, current_time_sec: float) -> None:
        target_value = self.samples[-1][1]
        alpha = min(1.0, dt * 6.0)
        self.display_latest += (target_value - self.display_latest) * alpha

        cutoff = current_time_sec - GRAPH_TIME_WINDOW_SEC
        visible = [v for (t, v) in self.samples if t >= cutoff]
        if not visible:
            visible = [target_value]
        target_max = max(5.0, max(max(visible), self.display_latest) * 1.15)
        max_alpha = min(1.0, dt * 3.0)
        self.display_y_max += (target_max - self.display_y_max) * max_alpha

    def draw(
        self,
        surface: pygame.Surface,
        title_font: pygame.font.Font,
        small_font: pygame.font.Font,
        current_time_sec: float,
    ) -> None:
        pygame.draw.rect(surface, GRAPH_BG_COLOR, self.rect, border_radius=10)
        pygame.draw.rect(
            surface, PANEL_BORDER_COLOR, self.rect, width=1, border_radius=10
        )

        title = title_font.render(self.label, True, (225, 236, 230))
        value = small_font.render(
            f"Current: {int(round(self.samples[-1][1]))}", True, self.color
        )
        surface.blit(title, (self.rect.x + 12, self.rect.y + 8))
        surface.blit(value, (self.rect.x + 12, self.rect.y + 30))

        plot_rect = pygame.Rect(
            self.rect.x + 12,
            self.rect.y + 56,
            self.rect.width - 24,
            self.rect.height - 72,
        )

        # Axes and grid
        for i in range(5):
            gy = plot_rect.y + int(i * (plot_rect.height / 4))
            pygame.draw.line(
                surface, (45, 58, 61), (plot_rect.x, gy), (plot_rect.right, gy), 1
            )
        pygame.draw.rect(surface, (70, 86, 90), plot_rect, width=1)

        cutoff = current_time_sec - GRAPH_TIME_WINDOW_SEC

        # Build visible polyline points
        visible: list[tuple[float, float]] = []
        prev_sample: tuple[float, float] | None = None
        for sample in self.samples:
            if sample[0] < cutoff:
                prev_sample = sample
                continue
            if prev_sample is not None and not visible:
                # Add a boundary anchor for smoother entering line.
                visible.append((cutoff, prev_sample[1]))
            visible.append(sample)

        if not visible:
            visible = [(current_time_sec, self.samples[-1][1])]

        y_max = max(1.0, self.display_y_max)

        def to_screen(t: float, v: float) -> tuple[int, int]:
            tx = (t - cutoff) / GRAPH_TIME_WINDOW_SEC
            tx = max(0.0, min(1.0, tx))
            ty = max(0.0, min(1.0, v / y_max))
            x = plot_rect.x + int(tx * plot_rect.width)
            y = plot_rect.bottom - int(ty * plot_rect.height)
            return x, y

        if len(visible) >= 2:
            pts = [to_screen(t, v) for (t, v) in visible]
            pygame.draw.lines(surface, self.color, False, pts, 2)

        # Smooth latest dot at right edge
        dot_x = plot_rect.right
        dot_y = plot_rect.bottom - int(
            max(0.0, min(1.0, self.display_latest / y_max)) * plot_rect.height
        )
        pygame.draw.circle(surface, self.color, (dot_x, dot_y), 5)

        max_label = small_font.render(f"{int(round(y_max))}", True, (150, 165, 168))
        zero_label = small_font.render("0", True, (150, 165, 168))
        surface.blit(max_label, (plot_rect.x + 4, plot_rect.y + 2))
        surface.blit(zero_label, (plot_rect.x + 4, plot_rect.bottom - 16))


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

    total_width = WIDTH + PANEL_WIDTH
    screen = pygame.display.set_mode((total_width, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 15)

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

    panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, HEIGHT)
    world_rect = pygame.Rect(PANEL_WIDTH, 0, WIDTH, HEIGHT)

    margin = 14
    gap = 12
    graph_height = (HEIGHT - margin * 2 - gap) // 2
    sheep_graph = PopulationGraph(
        pygame.Rect(margin, margin, PANEL_WIDTH - margin * 2, graph_height),
        "Sheep Population",
        SHEEP_GRAPH_COLOR,
        len(sheep_flock),
    )
    wolf_graph = PopulationGraph(
        pygame.Rect(
            margin, margin + graph_height + gap, PANEL_WIDTH - margin * 2, graph_height
        ),
        "Wolf Population",
        WOLF_GRAPH_COLOR,
        len(wolf_pack),
    )

    sim_time = 0.0
    sample_accum = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt
        sample_accum += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        all_animals: list[Sheep | Wolf] = [*sheep_flock, *wolf_pack]

        for animal in all_animals:
            animal.update(dt)

        resolve_interactions(sheep_flock, wolf_pack, sheep_animation_frames)

        # Feed graph history on a fixed cadence to keep lines readable.
        while sample_accum >= GRAPH_SAMPLE_INTERVAL_SEC:
            sample_accum -= GRAPH_SAMPLE_INTERVAL_SEC
            sheep_graph.add_sample(sim_time, len(sheep_flock))
            wolf_graph.add_sample(sim_time, len(wolf_pack))

        sheep_graph.update(dt, sim_time)
        wolf_graph.update(dt, sim_time)

        all_animals = [*sheep_flock, *wolf_pack]

        # Draw split layout
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, PANEL_BG_COLOR, panel_rect)
        pygame.draw.rect(screen, BG_COLOR, world_rect)
        pygame.draw.line(
            screen,
            PANEL_BORDER_COLOR,
            (PANEL_WIDTH, 0),
            (PANEL_WIDTH, HEIGHT),
            2,
        )

        sheep_graph.draw(screen, font, small_font, sim_time)
        wolf_graph.draw(screen, font, small_font, sim_time)

        for animal in all_animals:
            animal.draw(screen, x_offset=PANEL_WIDTH)

        fps_text = small_font.render(
            f"FPS: {clock.get_fps():5.1f}", True, (220, 230, 225)
        )
        screen.blit(fps_text, (12, HEIGHT - 24))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
