import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pygame
from pygame.math import Vector2
from motors import MovementMotor, StraightLineMotor, RandomWalkMotor


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
# ID Generator
# ------------------------------------------------------------
class UniqueIdGenerator:
    def __init__(self, start: int = 1):
        self._next_id = max(1, start)

    def next_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value


# ------------------------------------------------------------
# Animals
# ------------------------------------------------------------
class Sheep:
    def __init__(
        self,
        animal_id: int,
        motor: MovementMotor,
        position: Vector2,
        speed: float,
        scale: int,
        initial_reproduction_cooldown: float = 0.0,
    ):
        self.id = animal_id
        self.motor = motor
        self.speed = speed
        self.pos = position
        self.reproduction_cooldown = max(0.0, initial_reproduction_cooldown)
        self.is_alive = True

        initial_angle = random.uniform(0.0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )
        self.base_radius = scale * 0.38

    def move(self, dt: float, displacement_scale: float) -> None:
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

        if self.reproduction_cooldown > 0.0:
            self.reproduction_cooldown = max(0.0, self.reproduction_cooldown - dt)

    def eat(self) -> None:
        # Sheep currently do not eat other agents.
        return

    def die(self) -> None:
        self.is_alive = False

    def can_reproduce(self) -> bool:
        return self.reproduction_cooldown <= 0.0 and self.is_alive

    def reproduce(
        self,
        other: "Sheep",
        child_id: int,
        child_motor: MovementMotor,
        child_speed: float,
        child_scale: int,
        cooldown_sec: float,
    ) -> "Sheep | None":
        if not self.can_reproduce() or not other.can_reproduce():
            return None

        midpoint = (self.pos + other.pos) * 0.5
        offset = Vector2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        if offset.length_squared() < 1e-6:
            offset = Vector2(1.0, 0.0)
        offset = offset.normalize() * random.uniform(0.0, self.base_radius)

        child = Sheep(
            animal_id=child_id,
            motor=child_motor,
            position=midpoint + offset,
            speed=child_speed,
            scale=child_scale,
            initial_reproduction_cooldown=cooldown_sec,
        )

        self.reproduction_cooldown = cooldown_sec
        other.reproduction_cooldown = cooldown_sec
        return child


class Wolf:
    def __init__(
        self,
        animal_id: int,
        motor: MovementMotor,
        position: Vector2,
        speed: float,
        scale: int,
    ):
        self.id = animal_id
        self.motor = motor
        self.speed = speed
        self.pos = position
        self.is_alive = True

        initial_angle = random.uniform(0.0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )
        self.base_radius = scale * 0.38

    def move(self, dt: float, displacement_scale: float) -> None:
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

    def eat(self, target_sheep: Sheep, sheep_by_id: dict[int, Sheep]) -> None:
        if target_sheep.id in sheep_by_id:
            target_sheep.die()
            del sheep_by_id[target_sheep.id]

    def die(self) -> None:
        self.is_alive = False

    def reproduce(self) -> None:
        # Placeholder for future wolf reproduction behavior.
        return


Agent = Sheep | Wolf


# ------------------------------------------------------------
# Collision Contracts
# ------------------------------------------------------------
class CollidableAnimal(Protocol):
    id: int
    pos: Vector2
    vel: Vector2
    base_radius: float
    speed: float


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

        if b.vel.length_squared() > 1e-6:
            b.vel = b.vel.normalize() * b.speed


# ------------------------------------------------------------
# Painter
# ------------------------------------------------------------
@dataclass
class AnimalVisual:
    animation_frames: list[pygame.Surface]
    step_expand: float
    step_compress: float
    turn_duration_sec: float
    anim_fps: float
    anim_frame_cursor: float
    display_angle: float
    target_angle: float
    turn_start_angle: float
    turn_elapsed: float


class Painter:
    def __init__(
        self,
        sheep_animation_frames: list[pygame.Surface],
        wolf_animation_frames: list[pygame.Surface],
    ):
        self.sheep_animation_frames = sheep_animation_frames
        self.wolf_animation_frames = wolf_animation_frames
        self.visuals_by_id: dict[int, AnimalVisual] = {}

    @staticmethod
    def _velocity_to_display_angle(vel: Vector2) -> float:
        return math.degrees(math.atan2(-vel.x, vel.y))

    def _get_species_visual_setup(
        self, animal: Agent
    ) -> tuple[list[pygame.Surface], float, float]:
        if isinstance(animal, Sheep):
            return (
                self.sheep_animation_frames,
                SHEEP_STEP_SPEED_MULT_EXPAND,
                SHEEP_STEP_SPEED_MULT_COMPRESS,
            )
        return (
            self.wolf_animation_frames,
            WOLF_STEP_SPEED_MULT_EXPAND,
            WOLF_STEP_SPEED_MULT_COMPRESS,
        )

    def _ensure_visual(self, animal: Agent) -> AnimalVisual:
        visual = self.visuals_by_id.get(animal.id)
        if visual is not None:
            return visual

        frames, step_expand, step_compress = self._get_species_visual_setup(animal)
        initial_angle = self._velocity_to_display_angle(animal.vel)
        visual = AnimalVisual(
            animation_frames=frames,
            step_expand=step_expand,
            step_compress=step_compress,
            turn_duration_sec=TURN_DURATION_SEC,
            anim_fps=ANIM_FPS,
            anim_frame_cursor=random.uniform(0.0, float(len(frames))),
            display_angle=initial_angle,
            target_angle=initial_angle,
            turn_start_angle=initial_angle,
            turn_elapsed=TURN_DURATION_SEC,
        )
        self.visuals_by_id[animal.id] = visual
        return visual

    def get_displacement_scale(self, animal: Agent) -> float:
        visual = self._ensure_visual(animal)
        if animal.vel.length_squared() < 1e-6:
            return 0.0

        frame_count = len(visual.animation_frames)
        if frame_count <= 0:
            return 1.0

        cycle_phase = (visual.anim_frame_cursor % frame_count) / frame_count
        if cycle_phase < 0.5:
            return visual.step_expand
        return visual.step_compress

    def update_visual(self, animal: Agent, dt: float) -> None:
        visual = self._ensure_visual(animal)

        if animal.vel.length_squared() > 1e-6:
            new_target = self._velocity_to_display_angle(animal.vel)
            if abs(angle_diff_deg(visual.target_angle, new_target)) > 1e-3:
                visual.turn_start_angle = visual.display_angle
                visual.target_angle = new_target
                visual.turn_elapsed = 0.0

        if visual.turn_duration_sec <= 1e-6:
            visual.display_angle = visual.target_angle
            visual.turn_elapsed = visual.turn_duration_sec
        elif visual.turn_elapsed < visual.turn_duration_sec:
            visual.turn_elapsed = min(
                visual.turn_duration_sec, visual.turn_elapsed + dt
            )
            progress = visual.turn_elapsed / visual.turn_duration_sec
            delta = angle_diff_deg(visual.turn_start_angle, visual.target_angle)
            visual.display_angle = visual.turn_start_angle + delta * progress
        else:
            visual.display_angle = visual.target_angle

        if animal.vel.length_squared() > 1e-6 and len(visual.animation_frames) > 0:
            visual.anim_frame_cursor = (
                visual.anim_frame_cursor + visual.anim_fps * dt
            ) % len(visual.animation_frames)

    def draw_agent(
        self, screen: pygame.Surface, animal: Agent, x_offset: int = 0
    ) -> None:
        visual = self._ensure_visual(animal)

        if len(visual.animation_frames) == 0:
            return

        frame_index = int(visual.anim_frame_cursor) % len(visual.animation_frames)
        base_image = visual.animation_frames[frame_index]

        render_angle = -visual.display_angle
        rotated = pygame.transform.rotozoom(base_image, render_angle, 1.0)
        rect = rotated.get_rect(center=(animal.pos.x + x_offset, animal.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = Vector2(animal.pos.x + x_offset, animal.pos.y)
        line_end = line_start + direction * (animal.base_radius + 18)
        pygame.draw.line(
            screen,
            (245, 245, 245),
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            2,
        )

    def draw_agents(
        self,
        screen: pygame.Surface,
        sheep_by_id: dict[int, Sheep],
        wolf_by_id: dict[int, Wolf],
        x_offset: int = 0,
    ) -> None:
        for sheep in sheep_by_id.values():
            self.draw_agent(screen, sheep, x_offset)
        for wolf in wolf_by_id.values():
            self.draw_agent(screen, wolf, x_offset)

    def sync_live_ids(self, live_ids: set[int]) -> None:
        stale_ids = [aid for aid in self.visuals_by_id.keys() if aid not in live_ids]
        for aid in stale_ids:
            self.visuals_by_id.pop(aid, None)


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

    id_generator = UniqueIdGenerator(start=1)

    sheep_by_id: dict[int, Sheep] = {}
    for _ in range(NUM_SHEEP):
        sid = id_generator.next_id()
        sheep_by_id[sid] = Sheep(
            animal_id=sid,
            motor=RandomWalkMotor(),
            position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
            speed=SHEEP_SPEED,
            scale=SHEEP_SCALE,
        )

    wolf_by_id: dict[int, Wolf] = {}
    for _ in range(NUM_WOLVES):
        wid = id_generator.next_id()
        wolf_by_id[wid] = Wolf(
            animal_id=wid,
            motor=StraightLineMotor(),
            position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
            speed=WOLF_SPEED,
            scale=WOLF_SCALE,
        )

    # Placeholder dictionary for future grass entities.
    grass_by_id: dict[int, object] = {}

    painter = Painter(sheep_animation_frames, wolf_animation_frames)

    panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, HEIGHT)
    world_rect = pygame.Rect(PANEL_WIDTH, 0, WIDTH, HEIGHT)

    margin = 14
    gap = 12
    graph_height = (HEIGHT - margin * 2 - gap) // 2
    sheep_graph = PopulationGraph(
        pygame.Rect(margin, margin, PANEL_WIDTH - margin * 2, graph_height),
        "Sheep Population",
        SHEEP_GRAPH_COLOR,
        len(sheep_by_id),
    )
    wolf_graph = PopulationGraph(
        pygame.Rect(
            margin, margin + graph_height + gap, PANEL_WIDTH - margin * 2, graph_height
        ),
        "Wolf Population",
        WOLF_GRAPH_COLOR,
        len(wolf_by_id),
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

        # 1) Move all logical agents.
        all_animals: list[Agent] = [*sheep_by_id.values(), *wolf_by_id.values()]
        for animal in all_animals:
            displacement_scale = painter.get_displacement_scale(animal)
            animal.move(dt, displacement_scale)
            animal.pos, animal.vel = bounce_in_bounds(
                animal.pos, animal.vel, animal.base_radius, WIDTH, HEIGHT
            )
            painter.update_visual(animal, dt)

        # 2) Pairwise interactions (no global resolve_interactions function).
        all_animals = [*sheep_by_id.values(), *wolf_by_id.values()]
        newborn_sheep_by_id: dict[int, Sheep] = {}

        for i in range(len(all_animals)):
            a = all_animals[i]
            if isinstance(a, Sheep) and a.id not in sheep_by_id:
                continue

            for j in range(i + 1, len(all_animals)):
                b = all_animals[j]
                if isinstance(b, Sheep) and b.id not in sheep_by_id:
                    continue

                delta = b.pos - a.pos
                min_dist = a.base_radius + b.base_radius
                dist_sq = delta.length_squared()
                if dist_sq >= min_dist * min_dist:
                    continue

                # Wolf eats sheep. Deletion happens inside Wolf.eat().
                if isinstance(a, Wolf) and isinstance(b, Sheep):
                    a.eat(b, sheep_by_id)
                    continue
                if isinstance(a, Sheep) and isinstance(b, Wolf):
                    b.eat(a, sheep_by_id)
                    continue

                if dist_sq < 1e-12:
                    angle = random.uniform(0.0, math.tau)
                    normal = Vector2(math.cos(angle), math.sin(angle))
                    dist = 0.0
                else:
                    dist = math.sqrt(dist_sq)
                    normal = delta / dist

                # Sheep reproduction.
                if isinstance(a, Sheep) and isinstance(b, Sheep):
                    if len(sheep_by_id) + len(newborn_sheep_by_id) < MAX_SHEEP:
                        child = a.reproduce(
                            other=b,
                            child_id=id_generator.next_id(),
                            child_motor=RandomWalkMotor(),
                            child_speed=SHEEP_SPEED,
                            child_scale=SHEEP_SCALE,
                            cooldown_sec=SHEEP_REPRODUCTION_COOLDOWN_SEC,
                        )
                        if child is not None:
                            child.pos, child.vel = bounce_in_bounds(
                                child.pos,
                                child.vel,
                                child.base_radius,
                                WIDTH,
                                HEIGHT,
                            )
                            newborn_sheep_by_id[child.id] = child

                elastic_collision_response(a, b, normal, dist, min_dist)

        if newborn_sheep_by_id:
            sheep_by_id.update(newborn_sheep_by_id)

        live_ids = set(sheep_by_id.keys()) | set(wolf_by_id.keys())
        painter.sync_live_ids(live_ids)

        # Keep variable alive for future grass system integration.
        _ = grass_by_id

        # Feed graph history on a fixed cadence to keep lines readable.
        while sample_accum >= GRAPH_SAMPLE_INTERVAL_SEC:
            sample_accum -= GRAPH_SAMPLE_INTERVAL_SEC
            sheep_graph.add_sample(sim_time, len(sheep_by_id))
            wolf_graph.add_sample(sim_time, len(wolf_by_id))

        sheep_graph.update(dt, sim_time)
        wolf_graph.update(dt, sim_time)

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

        painter.draw_agents(screen, sheep_by_id, wolf_by_id, x_offset=PANEL_WIDTH)

        fps_text = small_font.render(
            f"FPS: {clock.get_fps():5.1f}", True, (220, 230, 225)
        )
        screen.blit(fps_text, (12, HEIGHT - 24))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
