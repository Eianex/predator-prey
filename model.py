import numpy as np
from numba_collision_kernel import solve_collisions_same_species
from numba_search_kernel import (
    collect_nearby_indices,
    nearest_alive_index,
    nearest_alive_grown_index,
)
import functools
import math
import random
import time
from collections import deque
from pathlib import Path

from pygame.math import Vector2

from motors import (
    MovementMotor,
    RandomWalkMotor,
    StraightLineMotor,
    TargetStraightMotor,
)
from ui import SimulationGUI
from save_csv import PopulationRecorder


class FunctionProfiler:
    def __init__(self, target_fps: float, window_size: int = 5):
        self.target_fps = float(target_fps)
        self.window_size = max(1, int(window_size))
        self.current_frame_time_by_name: dict[str, float] = {}
        self.recent_frames: deque[dict[str, float]] = deque(maxlen=self.window_size)

    def record(self, name: str, elapsed_sec: float) -> None:
        self.current_frame_time_by_name[name] = (
            self.current_frame_time_by_name.get(name, 0.0) + elapsed_sec
        )

    def end_frame(self) -> None:
        self.recent_frames.append(self.current_frame_time_by_name)
        self.current_frame_time_by_name = {}

    def format_hotspots(self, threshold_fps: float = 1.0) -> str:
        frame_count = len(self.recent_frames)
        if frame_count <= 0:
            return ""

        total_time_by_name: dict[str, float] = {}
        for frame in self.recent_frames:
            for name, elapsed_sec in frame.items():
                total_time_by_name[name] = (
                    total_time_by_name.get(name, 0.0) + elapsed_sec
                )

        fps_scale = self.target_fps * self.target_fps / frame_count
        hotspots: list[tuple[str, float]] = []
        other_fps = 0.0

        for name, total_sec in total_time_by_name.items():
            cost_fps = total_sec * fps_scale
            if cost_fps >= threshold_fps:
                hotspots.append((name, cost_fps))
            else:
                other_fps += cost_fps

        hotspots.sort(key=lambda item: item[1], reverse=True)
        parts = [f"{name}: {cost_fps:4.1f}fps" for name, cost_fps in hotspots]
        if other_fps >= threshold_fps:
            parts.append(f"other<1fps: {other_fps:4.1f}fps")
        return " | ".join(parts)


def profile_method(label: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            profiler = getattr(self, "_profiler", None)
            if profiler is None:
                return func(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                profiler.record(label, time.perf_counter() - start)

        return wrapper

    return decorator


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIDTH, HEIGHT = 1000, 800
FPS = 60

ANIM_FRAME_COUNT = 120
ANIM_CYCLE_SEC = 0.5
ANIM_FPS = ANIM_FRAME_COUNT / ANIM_CYCLE_SEC
ANIMATION = False
SAVE_TO_FILE = False
HEADLESS = False

NUM_SHEEP = 100
NUM_WOLVES = 40
INITIAL_PLANTS = 250

MAX_SHEEP = 500
MAX_WOLVES = 500
MAX_GRASS = 500

SHEEP_SCALE = 20
WOLF_SCALE = 20
PLANT_SCALE = 20

SHEEP_SPEED = 40.0
WOLF_SPEED = 45.0

PLANT_GROWTH_SEC = 2.0
PLANT_REPRODUCTION_PERIOD_SEC = 2.0

SHEEP_NO_NEED_FOOD_SEC = 3
SHEEP_TIMER_TO_FIND_FOOD_SEC = 6.0
WOLF_NO_NEED_FOOD_SEC = 0.5
WOLF_TIMER_TO_FIND_FOOD_SEC = 2.6
WOLF_EAT_ALL = False
SHEEP_EAT_ALL = False

PLANT_REPRODUCTION_RADIUS = PLANT_SCALE
PLANT_NEARBY_RADIUS_MULT = 1.01
PLANT_NEARBY_LIMIT = 3
PLANT_RANDOM_SPAWN_CHANCE_PER_SEC = 1

SHEEP_TYPE_OF_REPRODUCTION = "asexual"
WOLF_TYPE_OF_REPRODUCTION = "asexual"
SHEEP_ASEXUAL_REPRODUCTION_DELAY_SEC = 0.30
SHEEP_ASEXUAL_REPRODUCTION_DELAY_JITTER_SEC = 0.15
WOLF_ASEXUAL_REPRODUCTION_DELAY_SEC = 0.30
WOLF_ASEXUAL_REPRODUCTION_DELAY_JITTER_SEC = 0.15
SHEEP_STEP_SPEED_MULT_EXPAND = 0.8
SHEEP_STEP_SPEED_MULT_COMPRESS = 2 - SHEEP_STEP_SPEED_MULT_EXPAND
WOLF_STEP_SPEED_MULT_EXPAND = 0.8
WOLF_STEP_SPEED_MULT_COMPRESS = 2 - WOLF_STEP_SPEED_MULT_EXPAND


GRAPH_SAMPLE_INTERVAL_SEC = 0.12


SIMULATION_CONTROL_SPECS = [
    {
        "key": "NUM_SHEEP",
        "label": "Initial sheep",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "NUM_WOLVES",
        "label": "Initial wolves",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "INITIAL_PLANTS",
        "label": "Initial grass",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "MAX_SHEEP",
        "label": "Max sheep",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "MAX_WOLVES",
        "label": "Max wolves",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "MAX_GRASS",
        "label": "Max grass",
        "minimum": 0.0,
        "maximum": 1000.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "SHEEP_SCALE",
        "label": "Sheep scale",
        "minimum": 1.0,
        "maximum": 100.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "WOLF_SCALE",
        "label": "Wolf scale",
        "minimum": 1.0,
        "maximum": 100.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "PLANT_SCALE",
        "label": "Plant scale",
        "minimum": 1.0,
        "maximum": 100.0,
        "step": 1.0,
        "integer": True,
        "decimals": 0,
    },
    {
        "key": "SHEEP_SPEED",
        "label": "Sheep speed",
        "minimum": 0.0,
        "maximum": 200.0,
        "step": 0.5,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "WOLF_SPEED",
        "label": "Wolf speed",
        "minimum": 0.0,
        "maximum": 200.0,
        "step": 0.5,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "PLANT_REPRODUCTION_PERIOD_SEC",
        "label": "Grass reproduction period [s]",
        "minimum": 0.0,
        "maximum": 5.0,
        "step": 0.1,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "SHEEP_NO_NEED_FOOD_SEC",
        "label": "Sheep no need food [s]",
        "minimum": 0.0,
        "maximum": 20.0,
        "step": 0.1,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "SHEEP_TIMER_TO_FIND_FOOD_SEC",
        "label": "Sheep timer to find food [s]",
        "minimum": 0.0,
        "maximum": 20.0,
        "step": 0.1,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "WOLF_NO_NEED_FOOD_SEC",
        "label": "Wolf no need food [s]",
        "minimum": 0.0,
        "maximum": 20.0,
        "step": 0.1,
        "integer": False,
        "decimals": 1,
    },
    {
        "key": "WOLF_TIMER_TO_FIND_FOOD_SEC",
        "label": "Wolf timer to find food [s]",
        "minimum": 0.0,
        "maximum": 20.0,
        "step": 0.1,
        "integer": False,
        "decimals": 1,
    },
]

SIMULATION_TOGGLE_SPECS = [
    {
        "key": "WOLF_EAT_ALL",
        "label": "Wolf unlimited eating",
    },
    {
        "key": "SHEEP_EAT_ALL",
        "label": "Sheep unlimited eating",
    },
]

SETTINGS_FILE_PATH = Path("settings.txt")


def default_simulation_settings() -> dict[str, float | bool]:
    return {
        "NUM_SHEEP": float(NUM_SHEEP),
        "NUM_WOLVES": float(NUM_WOLVES),
        "INITIAL_PLANTS": float(INITIAL_PLANTS),
        "MAX_SHEEP": float(MAX_SHEEP),
        "MAX_WOLVES": float(MAX_WOLVES),
        "MAX_GRASS": float(MAX_GRASS),
        "SHEEP_SCALE": float(SHEEP_SCALE),
        "WOLF_SCALE": float(WOLF_SCALE),
        "PLANT_SCALE": float(PLANT_SCALE),
        "SHEEP_SPEED": float(SHEEP_SPEED),
        "WOLF_SPEED": float(WOLF_SPEED),
        "PLANT_GROWTH_SEC": float(PLANT_GROWTH_SEC),
        "PLANT_REPRODUCTION_PERIOD_SEC": float(PLANT_REPRODUCTION_PERIOD_SEC),
        "SHEEP_NO_NEED_FOOD_SEC": float(SHEEP_NO_NEED_FOOD_SEC),
        "SHEEP_TIMER_TO_FIND_FOOD_SEC": float(SHEEP_TIMER_TO_FIND_FOOD_SEC),
        "WOLF_NO_NEED_FOOD_SEC": float(WOLF_NO_NEED_FOOD_SEC),
        "WOLF_TIMER_TO_FIND_FOOD_SEC": float(WOLF_TIMER_TO_FIND_FOOD_SEC),
        "WOLF_EAT_ALL": WOLF_EAT_ALL,
        "SHEEP_EAT_ALL": SHEEP_EAT_ALL,
    }


def _parse_bool_setting(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {raw_value}")


def save_simulation_settings(settings: dict[str, float | bool]) -> None:
    defaults = default_simulation_settings()
    lines: list[str] = []
    for spec in SIMULATION_CONTROL_SPECS:
        key = spec["key"]
        value = float(settings.get(key, defaults[key]))
        if spec.get("integer", False):
            formatted = str(int(round(value)))
        else:
            formatted = f"{value:.{int(spec.get('decimals', 2))}f}"
        lines.append(f"{key}={formatted}")

    for spec in SIMULATION_TOGGLE_SPECS:
        key = spec["key"]
        formatted = "true" if bool(settings.get(key, defaults[key])) else "false"
        lines.append(f"{key}={formatted}")

    SETTINGS_FILE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved settings to {SETTINGS_FILE_PATH}")


def load_simulation_settings() -> dict[str, float | bool] | None:
    if not SETTINGS_FILE_PATH.exists():
        print(f"Settings file not found: {SETTINGS_FILE_PATH}")
        return None

    settings = default_simulation_settings()
    for raw_line in SETTINGS_FILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in settings:
            continue

        default_value = settings[key]
        try:
            if isinstance(default_value, bool):
                settings[key] = _parse_bool_setting(raw_value)
            else:
                settings[key] = float(raw_value)
        except ValueError:
            print(f"Ignoring invalid value for {key}: {raw_value}")

    print(f"Imported settings from {SETTINGS_FILE_PATH}")
    return settings


def apply_simulation_settings(settings: dict[str, float | bool]) -> None:
    global NUM_SHEEP
    global NUM_WOLVES
    global INITIAL_PLANTS
    global MAX_SHEEP
    global MAX_WOLVES
    global MAX_GRASS
    global SHEEP_SCALE
    global WOLF_SCALE
    global PLANT_SCALE
    global SHEEP_SPEED
    global WOLF_SPEED
    global PLANT_GROWTH_SEC
    global PLANT_REPRODUCTION_PERIOD_SEC
    global PLANT_NEARBY_LIMIT
    global PLANT_RANDOM_SPAWN_CHANCE_PER_SEC
    global SHEEP_NO_NEED_FOOD_SEC
    global SHEEP_TIMER_TO_FIND_FOOD_SEC
    global WOLF_NO_NEED_FOOD_SEC
    global WOLF_TIMER_TO_FIND_FOOD_SEC
    global WOLF_EAT_ALL
    global SHEEP_EAT_ALL

    NUM_SHEEP = max(0, int(round(settings["NUM_SHEEP"])))
    NUM_WOLVES = max(0, int(round(settings["NUM_WOLVES"])))
    INITIAL_PLANTS = max(0, int(round(settings["INITIAL_PLANTS"])))
    MAX_SHEEP = max(NUM_SHEEP, int(round(settings["MAX_SHEEP"])))
    MAX_WOLVES = max(NUM_WOLVES, int(round(settings["MAX_WOLVES"])))
    MAX_GRASS = max(INITIAL_PLANTS, int(round(settings["MAX_GRASS"])))
    SHEEP_SCALE = max(1, int(round(settings["SHEEP_SCALE"])))
    WOLF_SCALE = max(1, int(round(settings["WOLF_SCALE"])))
    PLANT_SCALE = max(1, int(round(settings["PLANT_SCALE"])))
    SHEEP_SPEED = max(0.0, float(settings["SHEEP_SPEED"]))
    WOLF_SPEED = max(0.0, float(settings["WOLF_SPEED"]))
    PLANT_GROWTH_SEC = max(
        0.0, float(settings.get("PLANT_GROWTH_SEC", PLANT_GROWTH_SEC))
    )
    PLANT_REPRODUCTION_PERIOD_SEC = max(
        0.0, float(settings["PLANT_REPRODUCTION_PERIOD_SEC"])
    )
    SHEEP_NO_NEED_FOOD_SEC = max(0.0, float(settings["SHEEP_NO_NEED_FOOD_SEC"]))
    SHEEP_TIMER_TO_FIND_FOOD_SEC = max(
        0.0, float(settings["SHEEP_TIMER_TO_FIND_FOOD_SEC"])
    )
    WOLF_NO_NEED_FOOD_SEC = max(0.0, float(settings["WOLF_NO_NEED_FOOD_SEC"]))
    WOLF_TIMER_TO_FIND_FOOD_SEC = max(
        0.0, float(settings["WOLF_TIMER_TO_FIND_FOOD_SEC"])
    )
    WOLF_EAT_ALL = bool(settings.get("WOLF_EAT_ALL", WOLF_EAT_ALL))
    SHEEP_EAT_ALL = bool(settings.get("SHEEP_EAT_ALL", SHEEP_EAT_ALL))


# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------
class PendingAsexualBirth:
    def __init__(self, species: str, parent_id: int, delay_sec: float):
        self.species = species
        self.parent_id = parent_id
        self.delay_sec = max(0.0, delay_sec)


class AsexualReproduction:
    @staticmethod
    def spawn_position(origin: Vector2, radius: float) -> Vector2:
        offset = Vector2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        if offset.length_squared() < 1e-6:
            offset = Vector2(1.0, 0.0)
        offset = offset.normalize() * random.uniform(0.0, radius)
        return origin + offset

    @staticmethod
    def delay_seconds(base_sec: float, jitter_sec: float) -> float:
        return max(0.0, base_sec + random.uniform(-jitter_sec, jitter_sec))


class Sheep:
    def __init__(
        self,
        animal_id: int,
        motor: MovementMotor,
        position: Vector2,
        speed: float,
        scale: int,
        no_need_food_sec: float | None = None,
        timer_to_find_food_sec: float | None = None,
        initial_food_offset_sec: float | None = None,
    ):
        self.id = animal_id
        self.motor = motor
        self.speed = speed
        self.pos = position

        if no_need_food_sec is None:
            no_need_food_sec = SHEEP_NO_NEED_FOOD_SEC
        if timer_to_find_food_sec is None:
            timer_to_find_food_sec = SHEEP_TIMER_TO_FIND_FOOD_SEC

        self.no_need_food_sec = max(0.0, no_need_food_sec)
        self.timer_to_find_food_sec = max(0.0, timer_to_find_food_sec)

        self.no_need_food_timer = self.no_need_food_sec
        self.find_food_timer = self.timer_to_find_food_sec
        self.is_alive = True

        initial_angle = random.uniform(0.0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )
        self.base_radius = scale * 0.38
        self.motion_frame = random.uniform(0.0, float(ANIM_FRAME_COUNT))
        self.target_grass_id: int | None = None
        self.reset_food_cycle(initial_food_offset_sec)

    def reset_food_cycle(self, initial_offset_sec: float | None = None) -> None:
        total_cycle = self.no_need_food_sec + self.timer_to_find_food_sec
        if initial_offset_sec is None:
            self.no_need_food_timer = self.no_need_food_sec
            self.find_food_timer = self.timer_to_find_food_sec
            return

        offset = max(0.0, min(total_cycle, initial_offset_sec))
        if offset < self.no_need_food_sec:
            self.no_need_food_timer = self.no_need_food_sec - offset
            self.find_food_timer = self.timer_to_find_food_sec
        else:
            self.no_need_food_timer = 0.0
            self.find_food_timer = max(0.0, total_cycle - offset)

    def move(self, dt: float, displacement_scale: float) -> None:
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

    def act(self, world: "World", dt: float) -> None:
        if SHEEP_EAT_ALL and self.try_eat_grass(world):
            return
        if not self._can_search_food(world, dt):
            world.clear_sheep_grass_target(self)
            return
        self.try_acquire_grass_target(world)
        self.try_eat_grass(world)

    def _can_search_food(self, world: "World", dt: float) -> bool:
        if self.id in world.pending_dead_ids:
            return False
        if self.no_need_food_timer > 0.0:
            self.no_need_food_timer = max(0.0, self.no_need_food_timer - dt)
            return False

        self.find_food_timer = max(0.0, self.find_food_timer - dt)
        if self.find_food_timer <= 0.0:
            world.mark_dead(self)
            return False
        return True

    def try_acquire_grass_target(self, world: "World") -> None:
        if self.id in world.pending_dead_ids:
            return
        if not isinstance(self.motor, TargetStraightMotor):
            return

        world.validate_sheep_grass_target(self)

        if self.target_grass_id is not None:
            target = world.grass_by_id.get(self.target_grass_id)
            if target is None or target.id in world.pending_dead_ids:
                world.clear_sheep_grass_target(self)
                return
            if not self.motor.target_acquired:
                self.motor.set_target(target.pos)
            return

        nearest_plant = world.get_nearest_grass_entity(self.pos)
        if nearest_plant is None:
            self.motor.clear_target()
            return

        if world.claim_grass_for_sheep(self, nearest_plant):
            self.motor.set_target(nearest_plant.pos)

    def try_eat_grass(self, world: "World") -> bool:
        if self.id in world.pending_dead_ids:
            return False
        half_grass_size = PLANT_SCALE * 0.5
        search_radius = self.base_radius + (half_grass_size * math.sqrt(2.0))
        nearby = world.get_nearby_grass(self.pos, search_radius)
        for plant in nearby:
            if plant.id in world.pending_dead_ids:
                continue
            if not plant.is_fully_grown():
                continue
            half = plant.scale * 0.5
            nearest_x = max(plant.pos.x - half, min(self.pos.x, plant.pos.x + half))
            nearest_y = max(plant.pos.y - half, min(self.pos.y, plant.pos.y + half))
            dx = self.pos.x - nearest_x
            dy = self.pos.y - nearest_y
            if (dx * dx + dy * dy) <= (self.base_radius * self.base_radius):
                self.eat(plant, world)
                return True
        return False

    def eat(self, target_plant: "Plant", world: "World") -> None:
        world.mark_dead(target_plant)
        world.clear_sheep_grass_target(self)
        self.reset_food_cycle()

        if SHEEP_TYPE_OF_REPRODUCTION == "asexual":
            world.queue_asexual_birth(
                "sheep",
                self.id,
                AsexualReproduction.delay_seconds(
                    SHEEP_ASEXUAL_REPRODUCTION_DELAY_SEC,
                    SHEEP_ASEXUAL_REPRODUCTION_DELAY_JITTER_SEC,
                ),
            )

    def die(self) -> None:
        self.is_alive = False


class Wolf:
    def __init__(
        self,
        animal_id: int,
        motor: MovementMotor,
        position: Vector2,
        speed: float,
        scale: int,
        no_need_food_sec: float | None = None,
        timer_to_find_food_sec: float | None = None,
        initial_food_offset_sec: float | None = None,
    ):
        self.id = animal_id
        self.motor = motor
        self.speed = speed
        self.pos = position

        if no_need_food_sec is None:
            no_need_food_sec = WOLF_NO_NEED_FOOD_SEC
        if timer_to_find_food_sec is None:
            timer_to_find_food_sec = WOLF_TIMER_TO_FIND_FOOD_SEC

        self.no_need_food_sec = max(0.0, no_need_food_sec)
        self.timer_to_find_food_sec = max(0.0, timer_to_find_food_sec)

        self.no_need_food_timer = self.no_need_food_sec
        self.find_food_timer = self.timer_to_find_food_sec
        self.is_alive = True

        initial_angle = random.uniform(0.0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )
        self.base_radius = scale * 0.38
        self.motion_frame = random.uniform(0.0, float(ANIM_FRAME_COUNT))
        self.reset_food_cycle(initial_food_offset_sec)

    def reset_food_cycle(self, initial_offset_sec: float | None = None) -> None:
        total_cycle = self.no_need_food_sec + self.timer_to_find_food_sec
        if initial_offset_sec is None:
            self.no_need_food_timer = self.no_need_food_sec
            self.find_food_timer = self.timer_to_find_food_sec
            return

        offset = max(0.0, min(total_cycle, initial_offset_sec))
        if offset < self.no_need_food_sec:
            self.no_need_food_timer = self.no_need_food_sec - offset
            self.find_food_timer = self.timer_to_find_food_sec
        else:
            self.no_need_food_timer = 0.0
            self.find_food_timer = max(0.0, total_cycle - offset)

    def move(self, dt: float, displacement_scale: float) -> None:
        self.pos, self.vel = self.motor.advance(
            self.pos,
            self.vel,
            self.speed,
            dt,
            self.base_radius,
            displacement_scale,
        )

    def act(self, world: "World", dt: float) -> None:
        if WOLF_EAT_ALL and self.try_eat(world):
            return

        if not self._can_search_food(world, dt):
            if isinstance(self.motor, TargetStraightMotor):
                self.motor.clear_target()
            return

        if isinstance(self.motor, TargetStraightMotor):
            if not self.motor.target_acquired:
                nearest_sheep_pos = world.get_nearest_sheep(self.pos)
                if nearest_sheep_pos is not None:
                    self.motor.set_target(nearest_sheep_pos)
        self.try_eat(world)

    def _can_search_food(self, world: "World", dt: float) -> bool:
        if self.id in world.pending_dead_ids:
            return False
        if self.no_need_food_timer > 0.0:
            self.no_need_food_timer = max(0.0, self.no_need_food_timer - dt)
            return False

        self.find_food_timer = max(0.0, self.find_food_timer - dt)
        if self.find_food_timer <= 0.0:
            world.mark_dead(self)
            return False
        return True

    def try_eat(self, world: "World") -> bool:
        if self.id in world.pending_dead_ids:
            return False

        search_radius = self.base_radius + (SHEEP_SCALE * 0.38)
        nearby = world.get_nearby_sheep(self.pos, search_radius)
        for sheep in nearby:
            if sheep.id in world.pending_dead_ids:
                continue
            min_dist = self.base_radius + sheep.base_radius
            if (sheep.pos - self.pos).length_squared() < min_dist * min_dist:
                self.eat(sheep, world)
                return True
        return False

    def eat(self, target_sheep: Sheep, world: "World") -> None:
        world.mark_dead(target_sheep)
        if isinstance(self.motor, TargetStraightMotor):
            self.motor.clear_target()
        self.reset_food_cycle()

        if WOLF_TYPE_OF_REPRODUCTION == "asexual":
            world.queue_asexual_birth(
                "wolf",
                self.id,
                AsexualReproduction.delay_seconds(
                    WOLF_ASEXUAL_REPRODUCTION_DELAY_SEC,
                    WOLF_ASEXUAL_REPRODUCTION_DELAY_JITTER_SEC,
                ),
            )

    def die(self) -> None:
        self.is_alive = False


class Plant:
    def __init__(
        self,
        plant_id: int,
        position: Vector2,
        scale: int,
        initial_age_sec: float = 0.0,
    ):
        self.id = plant_id
        self.pos = position
        self.scale = scale
        self.age_sec = max(0.0, min(PLANT_GROWTH_SEC, initial_age_sec))
        self.reproduction_timer = PLANT_REPRODUCTION_PERIOD_SEC
        self.is_alive = True

    def is_fully_grown(self) -> bool:
        return self.age_sec >= PLANT_GROWTH_SEC

    def update(self, world: "World", dt: float) -> None:
        self.age_sec = min(PLANT_GROWTH_SEC, self.age_sec + dt)

        if not self.is_fully_grown():
            return

        self.reproduction_timer -= dt
        while self.reproduction_timer <= 0.0:
            self.reproduction_timer += PLANT_REPRODUCTION_PERIOD_SEC
            self.try_reproduce(world)

    def try_reproduce(self, world: "World") -> None:
        nearby_radius = PLANT_REPRODUCTION_RADIUS * PLANT_NEARBY_RADIUS_MULT
        if (
            world.count_nearby_grass(self.pos, nearby_radius, exclude=self.id)
            >= PLANT_NEARBY_LIMIT
        ):
            return

        nearby = world.get_nearby_grass(self.pos, nearby_radius, exclude=self.id)
        sum_x = 0.0
        sum_y = 0.0
        for plant in nearby:
            delta = plant.pos - self.pos
            if delta.length_squared() < 1e-12:
                continue
            angle = math.atan2(delta.y, delta.x)
            sum_x += math.cos(angle)
            sum_y += math.sin(angle)

        if abs(sum_x) < 1e-12 and abs(sum_y) < 1e-12:
            cluster_angle = 0.0
        else:
            cluster_angle = math.atan2(sum_y, sum_x)

        base_angle = (cluster_angle + math.pi) % math.tau
        spawn_angle = (
            base_angle + random.uniform(-math.pi / 2, math.pi / 2)
        ) % math.tau
        offset = (
            Vector2(math.cos(spawn_angle), math.sin(spawn_angle))
            * PLANT_REPRODUCTION_RADIUS
        )
        child = Plant(
            plant_id=world.allocate_id(),
            position=self.pos + offset,
            scale=PLANT_SCALE,
        )
        world.spawn_grass(child)

    def die(self) -> None:
        self.is_alive = False


Agent = Sheep | Wolf


# ------------------------------------------------------------
# World
# ------------------------------------------------------------
class World:
    def __init__(self):
        self._next_id = 1
        self.sheep_by_id: dict[int, Sheep] = {}
        self.wolf_by_id: dict[int, Wolf] = {}
        self.grass_by_id: dict[int, Plant] = {}

        self.pending_sheep_births: list[Sheep] = []
        self.pending_wolf_births: list[Wolf] = []
        self.pending_asexual_births: list[PendingAsexualBirth] = []
        self.pending_grass_births: list[Plant] = []
        self.pending_dead_ids: set[int] = set()
        self.grass_target_locks: dict[int, int] = {}
        self._profiler: FunctionProfiler | None = None

        self._sheep_refs: list[Sheep] = []
        self._sheep_ids = np.empty(0, dtype=np.int64)
        self._sheep_x = np.empty(0, dtype=np.float32)
        self._sheep_y = np.empty(0, dtype=np.float32)
        self._sheep_alive = np.empty(0, dtype=np.uint8)
        self._sheep_idx_by_id: dict[int, int] = {}

        self._grass_refs: list[Plant] = []
        self._grass_ids = np.empty(0, dtype=np.int64)
        self._grass_x = np.empty(0, dtype=np.float32)
        self._grass_y = np.empty(0, dtype=np.float32)
        self._grass_alive = np.empty(0, dtype=np.uint8)
        self._grass_grown = np.empty(0, dtype=np.uint8)
        self._grass_idx_by_id: dict[int, int] = {}

        target_initial_sheep = min(NUM_SHEEP, MAX_SHEEP)
        for _ in range(target_initial_sheep):
            sid = self.allocate_id()
            sheep = Sheep(
                animal_id=sid,
                motor=TargetStraightMotor(),
                position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                speed=SHEEP_SPEED,
                scale=SHEEP_SCALE,
                initial_food_offset_sec=self._random_initial_food_offset(
                    SHEEP_NO_NEED_FOOD_SEC, SHEEP_TIMER_TO_FIND_FOOD_SEC
                ),
            )
            self.sheep_by_id[sid] = sheep
            self._bounce_sheep_in_bounds(sheep)
            self._retarget_sheep_to_nearest_grass(sheep)

        self._rebuild_search_snapshots()

        target_initial_wolves = min(NUM_WOLVES, MAX_WOLVES)
        for _ in range(target_initial_wolves):
            wid = self.allocate_id()
            wolf = Wolf(
                animal_id=wid,
                motor=TargetStraightMotor(),
                position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                speed=WOLF_SPEED,
                scale=WOLF_SCALE,
                initial_food_offset_sec=self._random_initial_food_offset(
                    WOLF_NO_NEED_FOOD_SEC, WOLF_TIMER_TO_FIND_FOOD_SEC
                ),
            )
            self.wolf_by_id[wid] = wolf
            self._bounce_wolf_in_bounds(wolf)
            self._retarget_wolf_to_nearest_sheep(wolf)

        planted = 0
        attempts = 0
        target_initial_plants = min(INITIAL_PLANTS, MAX_GRASS)
        max_attempts = target_initial_plants * 20
        while planted < target_initial_plants and attempts < max_attempts:
            attempts += 1
            pid = self.allocate_id()
            plant = Plant(
                plant_id=pid,
                position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                scale=PLANT_SCALE,
                initial_age_sec=random.uniform(0.0, PLANT_GROWTH_SEC),
            )
            if self.can_place_grass(plant):
                self.grass_by_id[pid] = plant
                planted += 1

        self._rebuild_search_snapshots()

    def _rebuild_search_snapshots(self) -> None:
        sheep_refs = list(self.sheep_by_id.values())
        n_sheep = len(sheep_refs)

        self._sheep_refs = sheep_refs
        self._sheep_ids = np.empty(n_sheep, dtype=np.int64)
        self._sheep_x = np.empty(n_sheep, dtype=np.float32)
        self._sheep_y = np.empty(n_sheep, dtype=np.float32)
        self._sheep_alive = np.ones(n_sheep, dtype=np.uint8)
        self._sheep_idx_by_id = {}

        for i, sheep in enumerate(sheep_refs):
            self._sheep_ids[i] = sheep.id
            self._sheep_x[i] = sheep.pos.x
            self._sheep_y[i] = sheep.pos.y
            self._sheep_idx_by_id[sheep.id] = i

        grass_refs = list(self.grass_by_id.values())
        n_grass = len(grass_refs)

        self._grass_refs = grass_refs
        self._grass_ids = np.empty(n_grass, dtype=np.int64)
        self._grass_x = np.empty(n_grass, dtype=np.float32)
        self._grass_y = np.empty(n_grass, dtype=np.float32)
        self._grass_alive = np.ones(n_grass, dtype=np.uint8)
        self._grass_grown = np.empty(n_grass, dtype=np.uint8)
        self._grass_idx_by_id = {}

        for i, plant in enumerate(grass_refs):
            self._grass_ids[i] = plant.id
            self._grass_x[i] = plant.pos.x
            self._grass_y[i] = plant.pos.y
            self._grass_grown[i] = 1 if plant.is_fully_grown() else 0
            self._grass_idx_by_id[plant.id] = i

    def _start_wall_escape_if_on_boundary(self, animal: Agent) -> None:
        if not isinstance(animal.motor, TargetStraightMotor):
            return

        eps = 1e-4
        pad = 0.5
        escape = Vector2(0.0, 0.0)

        if animal.pos.x <= animal.base_radius + eps:
            animal.pos.x = animal.base_radius + pad
            escape.x += 1.0
        elif animal.pos.x >= WIDTH - animal.base_radius - eps:
            animal.pos.x = WIDTH - animal.base_radius - pad
            escape.x -= 1.0

        if animal.pos.y <= animal.base_radius + eps:
            animal.pos.y = animal.base_radius + pad
            escape.y += 1.0
        elif animal.pos.y >= HEIGHT - animal.base_radius - eps:
            animal.pos.y = HEIGHT - animal.base_radius - pad
            escape.y -= 1.0

        if escape.length_squared() > 1e-12:
            animal.motor.start_wall_escape(escape.normalize() * animal.speed, 0.12)

    @staticmethod
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

    @staticmethod
    def elastic_collision_response(
        a: Agent,
        b: Agent,
        normal: Vector2,
        dist: float,
        min_dist: float,
    ) -> None:
        overlap = min_dist - dist
        if overlap > 0.0:
            correction = normal * (overlap * 0.5)
            a.pos -= correction
            b.pos += correction

            a.pos, a.vel = World.bounce_in_bounds(
                a.pos, a.vel, a.base_radius, WIDTH, HEIGHT
            )
            b.pos, b.vel = World.bounce_in_bounds(
                b.pos, b.vel, b.base_radius, WIDTH, HEIGHT
            )

        va_n = a.vel.dot(normal)
        vb_n = b.vel.dot(normal)

        if va_n - vb_n > 0.0:
            a.vel += (vb_n - va_n) * normal
            b.vel += (va_n - vb_n) * normal

            if a.vel.length_squared() > 1e-6:
                a.vel = a.vel.normalize() * a.speed

            if b.vel.length_squared() > 1e-6:
                b.vel = b.vel.normalize() * b.speed

    def _solve_species_collisions(self, animals, retarget_fn) -> None:
        n = len(animals)
        if n < 2:
            return

        x = np.empty(n, dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        vx = np.empty(n, dtype=np.float32)
        vy = np.empty(n, dtype=np.float32)
        radius = np.empty(n, dtype=np.float32)
        speed = np.empty(n, dtype=np.float32)
        alive = np.empty(n, dtype=np.uint8)

        for i, a in enumerate(animals):
            x[i] = a.pos.x
            y[i] = a.pos.y
            vx[i] = a.vel.x
            vy[i] = a.vel.y
            radius[i] = a.base_radius
            speed[i] = a.speed
            alive[i] = 0 if a.id in self.pending_dead_ids else 1

        touched = solve_collisions_same_species(
            x, y, vx, vy, radius, speed, alive, WIDTH, HEIGHT
        )

        for i, a in enumerate(animals):
            a.pos.x = float(x[i])
            a.pos.y = float(y[i])
            a.vel.x = float(vx[i])
            a.vel.y = float(vy[i])
            self._start_wall_escape_if_on_boundary(a)

        touched_idx = np.nonzero(touched)[0]
        for i in touched_idx:
            retarget_fn(animals[int(i)])

    def allocate_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    @staticmethod
    def _random_initial_food_offset(
        no_need_food_sec: float, timer_to_find_food_sec: float
    ) -> float:
        total_cycle = no_need_food_sec + timer_to_find_food_sec
        if total_cycle <= 1e-6:
            return 0.0
        return random.uniform(0.0, max(0.0, total_cycle - 1e-6))

    def queue_asexual_birth(
        self, species: str, parent_id: int, delay_sec: float
    ) -> None:
        self.pending_asexual_births.append(
            PendingAsexualBirth(species, parent_id, delay_sec)
        )

    def _create_sheep_child_from_parent(self, parent: Sheep) -> Sheep | None:
        if not self.can_spawn_sheep():
            return None
        child_id = self.allocate_id()
        return Sheep(
            animal_id=child_id,
            motor=TargetStraightMotor(),
            position=AsexualReproduction.spawn_position(parent.pos, parent.base_radius),
            speed=SHEEP_SPEED,
            scale=SHEEP_SCALE,
        )

    def _create_wolf_child_from_parent(self, parent: Wolf) -> Wolf | None:
        if not self.can_spawn_wolf():
            return None
        child_id = self.allocate_id()
        return Wolf(
            animal_id=child_id,
            motor=TargetStraightMotor(),
            position=AsexualReproduction.spawn_position(parent.pos, parent.base_radius),
            speed=WOLF_SPEED,
            scale=WOLF_SCALE,
        )

    def _update_pending_asexual_births(self, dt: float) -> None:
        remaining: list[PendingAsexualBirth] = []
        for pending in self.pending_asexual_births:
            pending.delay_sec -= dt
            if pending.delay_sec > 0.0:
                remaining.append(pending)
                continue

            if pending.species == "sheep":
                parent = self.sheep_by_id.get(pending.parent_id)
                if parent is None or parent.id in self.pending_dead_ids:
                    continue
                child = self._create_sheep_child_from_parent(parent)
                if child is not None:
                    self.spawn_sheep(child)
                continue

            if pending.species == "wolf":
                parent = self.wolf_by_id.get(pending.parent_id)
                if parent is None or parent.id in self.pending_dead_ids:
                    continue
                child = self._create_wolf_child_from_parent(parent)
                if child is not None:
                    self.spawn_wolf(child)
                continue

        self.pending_asexual_births = remaining

    def all_agents(self) -> list[Agent]:
        return [*self.sheep_by_id.values(), *self.wolf_by_id.values()]

    def live_ids(self) -> set[int]:
        return (
            set(self.sheep_by_id.keys())
            | set(self.wolf_by_id.keys())
            | set(self.grass_by_id.keys())
        )

    def can_spawn_sheep(self) -> bool:
        return len(self.sheep_by_id) + len(self.pending_sheep_births) < MAX_SHEEP

    def can_spawn_wolf(self) -> bool:
        return len(self.wolf_by_id) + len(self.pending_wolf_births) < MAX_WOLVES

    def can_spawn_grass(self) -> bool:
        return len(self.grass_by_id) + len(self.pending_grass_births) < MAX_GRASS

    @staticmethod
    def _is_inside_bounds(pos: Vector2, half_size: float) -> bool:
        return (
            half_size <= pos.x <= WIDTH - half_size
            and half_size <= pos.y <= HEIGHT - half_size
        )

    def can_place_grass(self, plant: Plant, exclude: int | None = None) -> bool:
        half_size = plant.scale * 0.5
        if not self._is_inside_bounds(plant.pos, half_size):
            return False

        same_spot_dist_sq = 1.0

        for other in self.grass_by_id.values():
            if exclude is not None and other.id == exclude:
                continue
            if other.id in self.pending_dead_ids:
                continue
            if (other.pos - plant.pos).length_squared() <= same_spot_dist_sq:
                return False

        for pending in self.pending_grass_births:
            if exclude is not None and pending.id == exclude:
                continue
            if (pending.pos - plant.pos).length_squared() <= same_spot_dist_sq:
                return False

        return True

    def spawn_sheep(self, sheep: Sheep) -> None:
        if self.can_spawn_sheep():
            self._bounce_sheep_in_bounds(sheep)
            self._retarget_sheep_to_nearest_grass(sheep)
            self.pending_sheep_births.append(sheep)

    def spawn_wolf(self, wolf: Wolf) -> None:
        if self.can_spawn_wolf():
            self._bounce_wolf_in_bounds(wolf)
            self._retarget_wolf_to_nearest_sheep(wolf)
            self.pending_wolf_births.append(wolf)

    def spawn_grass(self, plant: Plant) -> None:
        if self.can_spawn_grass() and self.can_place_grass(plant):
            self.pending_grass_births.append(plant)

    def mark_dead(self, entity) -> None:
        if isinstance(entity, Plant):
            locked_by = self.grass_target_locks.pop(entity.id, None)
            if locked_by is not None:
                sheep = self.sheep_by_id.get(locked_by)
                if sheep is not None and sheep.target_grass_id == entity.id:
                    sheep.target_grass_id = None
                    if isinstance(sheep.motor, TargetStraightMotor):
                        sheep.motor.clear_target()

            idx = self._grass_idx_by_id.get(entity.id)
            if idx is not None:
                self._grass_alive[idx] = 0
                self._grass_grown[idx] = 0

        elif isinstance(entity, Sheep):
            self.clear_sheep_grass_target(entity)
            idx = self._sheep_idx_by_id.get(entity.id)
            if idx is not None:
                self._sheep_alive[idx] = 0

        self.pending_dead_ids.add(entity.id)

    def get_nearby_sheep(
        self, pos: Vector2, radius: float, exclude: int | None = None
    ) -> list[Sheep]:
        n = len(self._sheep_refs)
        if n == 0:
            return []

        out_idx = np.empty(n, dtype=np.int32)
        count = collect_nearby_indices(
            self._sheep_x,
            self._sheep_y,
            self._sheep_ids,
            self._sheep_alive,
            float(pos.x),
            float(pos.y),
            float(radius * radius),
            -1 if exclude is None else int(exclude),
            out_idx,
        )
        return [self._sheep_refs[int(out_idx[i])] for i in range(count)]

    def get_nearby_wolves(
        self, pos: Vector2, radius: float, exclude: int | None = None
    ) -> list[Wolf]:
        radius_sq = radius * radius
        nearby: list[Wolf] = []
        for wolf in self.wolf_by_id.values():
            if exclude is not None and wolf.id == exclude:
                continue
            if wolf.id in self.pending_dead_ids:
                continue
            if (wolf.pos - pos).length_squared() <= radius_sq:
                nearby.append(wolf)
        return nearby

    def get_nearby_grass(
        self, pos: Vector2, radius: float, exclude: int | None = None
    ) -> list[Plant]:
        n = len(self._grass_refs)
        if n == 0:
            return []

        out_idx = np.empty(n, dtype=np.int32)
        count = collect_nearby_indices(
            self._grass_x,
            self._grass_y,
            self._grass_ids,
            self._grass_alive,
            float(pos.x),
            float(pos.y),
            float(radius * radius),
            -1 if exclude is None else int(exclude),
            out_idx,
        )
        return [self._grass_refs[int(out_idx[i])] for i in range(count)]

    def get_nearest_sheep(
        self, pos: Vector2, exclude: int | None = None
    ) -> Vector2 | None:
        idx = nearest_alive_index(
            self._sheep_x,
            self._sheep_y,
            self._sheep_ids,
            self._sheep_alive,
            float(pos.x),
            float(pos.y),
            -1 if exclude is None else int(exclude),
        )
        if idx < 0:
            return None

        sheep = self._sheep_refs[int(idx)]
        return Vector2(sheep.pos.x, sheep.pos.y)

    def get_nearest_wolf(
        self, pos: Vector2, exclude: int | None = None
    ) -> Vector2 | None:
        nearest_pos: Vector2 | None = None
        nearest_dist_sq = float("inf")
        for wolf in self.wolf_by_id.values():
            if exclude is not None and wolf.id == exclude:
                continue
            if wolf.id in self.pending_dead_ids:
                continue
            dist_sq = (wolf.pos - pos).length_squared()
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest_pos = wolf.pos

        if nearest_pos is None:
            return None
        return Vector2(nearest_pos.x, nearest_pos.y)

    def get_nearest_grass(
        self, pos: Vector2, exclude: int | None = None
    ) -> Vector2 | None:
        nearest_pos: Vector2 | None = None
        nearest_dist_sq = float("inf")
        for plant in self.grass_by_id.values():
            if exclude is not None and plant.id == exclude:
                continue
            if plant.id in self.pending_dead_ids:
                continue
            dist_sq = (plant.pos - pos).length_squared()
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest_pos = plant.pos

        if nearest_pos is None:
            return None
        return Vector2(nearest_pos.x, nearest_pos.y)

    def get_nearest_grass_entity(
        self, pos: Vector2, exclude: int | None = None
    ) -> Plant | None:
        idx = nearest_alive_grown_index(
            self._grass_x,
            self._grass_y,
            self._grass_ids,
            self._grass_alive,
            self._grass_grown,
            float(pos.x),
            float(pos.y),
            -1 if exclude is None else int(exclude),
        )
        if idx < 0:
            return None
        return self._grass_refs[int(idx)]

    def clear_sheep_grass_target(self, sheep: Sheep) -> None:
        if sheep.target_grass_id is not None:
            locked_by = self.grass_target_locks.get(sheep.target_grass_id)
            if locked_by == sheep.id:
                self.grass_target_locks.pop(sheep.target_grass_id, None)
        sheep.target_grass_id = None
        if isinstance(sheep.motor, TargetStraightMotor):
            sheep.motor.clear_target()

    def validate_sheep_grass_target(self, sheep: Sheep) -> None:
        target_id = sheep.target_grass_id
        if target_id is None:
            return
        if target_id in self.pending_dead_ids:
            self.clear_sheep_grass_target(sheep)
            return
        plant = self.grass_by_id.get(target_id)
        if plant is None:
            self.clear_sheep_grass_target(sheep)
            return
        if not plant.is_fully_grown():
            self.clear_sheep_grass_target(sheep)
            return
        locked_by = self.grass_target_locks.get(target_id)
        if locked_by != sheep.id:
            self.clear_sheep_grass_target(sheep)

    def claim_grass_for_sheep(self, sheep: Sheep, plant: Plant) -> bool:
        if sheep.id in self.pending_dead_ids:
            return False
        if plant.id in self.pending_dead_ids:
            return False
        if not plant.is_fully_grown():
            return False
        locked_by = self.grass_target_locks.get(plant.id)
        if locked_by is not None and locked_by != sheep.id:
            return False

        if sheep.target_grass_id is not None and sheep.target_grass_id != plant.id:
            self.clear_sheep_grass_target(sheep)

        self.grass_target_locks[plant.id] = sheep.id
        sheep.target_grass_id = plant.id
        return True

    def _retarget_wolf_to_nearest_sheep(self, wolf: Wolf) -> None:
        if not isinstance(wolf.motor, TargetStraightMotor):
            return
        if wolf.no_need_food_timer > 0.0 or wolf.find_food_timer <= 0.0:
            wolf.motor.clear_target()
            return
        nearest_sheep_pos = self.get_nearest_sheep(wolf.pos)
        if nearest_sheep_pos is None:
            wolf.motor.clear_target()
        else:
            wolf.motor.set_target(nearest_sheep_pos)

    def _retarget_sheep_to_nearest_grass(self, sheep: Sheep) -> None:
        if not isinstance(sheep.motor, TargetStraightMotor):
            return
        self.validate_sheep_grass_target(sheep)
        if sheep.no_need_food_timer > 0.0 or sheep.find_food_timer <= 0.0:
            self.clear_sheep_grass_target(sheep)
            return
        if sheep.target_grass_id is not None:
            plant = self.grass_by_id.get(sheep.target_grass_id)
            if plant is None or plant.id in self.pending_dead_ids:
                self.clear_sheep_grass_target(sheep)
                return
            sheep.motor.set_target(plant.pos)
            return
        nearest_plant = self.get_nearest_grass_entity(sheep.pos)
        if nearest_plant is None:
            sheep.motor.clear_target()
            return
        if self.claim_grass_for_sheep(sheep, nearest_plant):
            sheep.motor.set_target(nearest_plant.pos)
        else:
            sheep.motor.clear_target()

    def _bounce_wolf_in_bounds(self, wolf: Wolf) -> bool:
        prev_pos = Vector2(wolf.pos.x, wolf.pos.y)
        prev_vel = Vector2(wolf.vel.x, wolf.vel.y)
        wolf.pos, wolf.vel = self.bounce_in_bounds(
            wolf.pos,
            wolf.vel,
            wolf.base_radius,
            WIDTH,
            HEIGHT,
        )
        pos_changed = (wolf.pos - prev_pos).length_squared() > 1e-12
        vel_changed = (wolf.vel - prev_vel).length_squared() > 1e-12
        if pos_changed or vel_changed:
            self._start_wall_escape_if_on_boundary(wolf)
            self._retarget_wolf_to_nearest_sheep(wolf)
            return True
        return False

    def _bounce_sheep_in_bounds(self, sheep: Sheep) -> bool:
        prev_pos = Vector2(sheep.pos.x, sheep.pos.y)
        prev_vel = Vector2(sheep.vel.x, sheep.vel.y)
        sheep.pos, sheep.vel = self.bounce_in_bounds(
            sheep.pos,
            sheep.vel,
            sheep.base_radius,
            WIDTH,
            HEIGHT,
        )
        pos_changed = (sheep.pos - prev_pos).length_squared() > 1e-12
        vel_changed = (sheep.vel - prev_vel).length_squared() > 1e-12
        if pos_changed or vel_changed:
            self._start_wall_escape_if_on_boundary(sheep)
            self._retarget_sheep_to_nearest_grass(sheep)
            return True
        return False

    def count_nearby_grass(
        self, pos: Vector2, radius: float, exclude: int | None = None
    ) -> int:
        radius_sq = radius * radius
        count = 0
        for plant in self.grass_by_id.values():
            if exclude is not None and plant.id == exclude:
                continue
            if plant.id in self.pending_dead_ids:
                continue
            if (plant.pos - pos).length_squared() <= radius_sq:
                count += 1

        for plant in self.pending_grass_births:
            if exclude is not None and plant.id == exclude:
                continue
            if (plant.pos - pos).length_squared() <= radius_sq:
                count += 1

        return count

    def _displacement_scale(self, agent: Agent) -> float:
        if agent.vel.length_squared() < 1e-6:
            return 0.0
        if not ANIMATION:
            return 1.0

        frame = agent.motion_frame
        cycle_phase = (frame % ANIM_FRAME_COUNT) / ANIM_FRAME_COUNT

        if isinstance(agent, Sheep):
            if cycle_phase < 0.5:
                return SHEEP_STEP_SPEED_MULT_EXPAND
            return SHEEP_STEP_SPEED_MULT_COMPRESS

        if cycle_phase < 0.5:
            return WOLF_STEP_SPEED_MULT_EXPAND
        return WOLF_STEP_SPEED_MULT_COMPRESS

    def _advance_motion_frame(self, agent: Agent, dt: float) -> None:
        if not ANIMATION:
            return
        if agent.vel.length_squared() < 1e-6:
            return
        agent.motion_frame = (agent.motion_frame + ANIM_FPS * dt) % ANIM_FRAME_COUNT

    def _move_agents(self, dt: float) -> None:
        for agent in self.all_agents():
            scale = self._displacement_scale(agent)
            agent.move(dt, scale)
            if isinstance(agent, Wolf):
                self._bounce_wolf_in_bounds(agent)
            else:
                self._bounce_sheep_in_bounds(agent)
            self._advance_motion_frame(agent, dt)

    def _resolve_physics_collisions(self) -> None:
        sheep = list(self.sheep_by_id.values())
        wolves = list(self.wolf_by_id.values())

        self._solve_species_collisions(sheep, self._retarget_sheep_to_nearest_grass)
        self._solve_species_collisions(wolves, self._retarget_wolf_to_nearest_sheep)

    def _act_agents(self, dt: float) -> None:
        for wolf in list(self.wolf_by_id.values()):
            if wolf.id in self.pending_dead_ids:
                continue
            wolf.act(self, dt)
            self._bounce_wolf_in_bounds(wolf)

        for sheep in list(self.sheep_by_id.values()):
            if sheep.id in self.pending_dead_ids:
                continue
            sheep.act(self, dt)
            self._bounce_sheep_in_bounds(sheep)

    def _update_grass(self, dt: float) -> None:
        for plant in list(self.grass_by_id.values()):
            if plant.id in self.pending_dead_ids:
                continue
            plant.update(self, dt)

    def _try_random_grass_spawn(self, dt: float) -> None:
        chance = PLANT_RANDOM_SPAWN_CHANCE_PER_SEC * dt
        if random.random() >= chance:
            return

        plant = Plant(
            plant_id=self.allocate_id(),
            position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
            scale=PLANT_SCALE,
        )
        self.spawn_grass(plant)

    def apply_pending_changes(self) -> None:
        for dead_id in list(self.pending_dead_ids):
            sheep = self.sheep_by_id.pop(dead_id, None)
            if sheep is not None:
                self.clear_sheep_grass_target(sheep)
                sheep.die()
            wolf = self.wolf_by_id.pop(dead_id, None)
            if wolf is not None:
                wolf.die()
            plant = self.grass_by_id.pop(dead_id, None)
            if plant is not None:
                self.grass_target_locks.pop(plant.id, None)
                plant.die()
        self.pending_dead_ids.clear()

        if self.pending_sheep_births:
            for child in self.pending_sheep_births:
                if child.id in self.sheep_by_id:
                    continue
                if len(self.sheep_by_id) >= MAX_SHEEP:
                    break
                self.sheep_by_id[child.id] = child
            self.pending_sheep_births.clear()

        if self.pending_wolf_births:
            for child in self.pending_wolf_births:
                if child.id in self.wolf_by_id:
                    continue
                if len(self.wolf_by_id) >= MAX_WOLVES:
                    break
                self.wolf_by_id[child.id] = child
            self.pending_wolf_births.clear()

        if self.pending_grass_births:
            for child in self.pending_grass_births:
                if child.id in self.grass_by_id:
                    continue
                if len(self.grass_by_id) >= MAX_GRASS:
                    break
                if self.can_place_grass(child, exclude=child.id):
                    self.grass_by_id[child.id] = child
            self.pending_grass_births.clear()

    def step(self, dt: float) -> None:
        self._move_agents(dt)
        self._resolve_physics_collisions()
        self._rebuild_search_snapshots()
        self._update_pending_asexual_births(dt)
        self._act_agents(dt)
        self._update_grass(dt)
        self._try_random_grass_spawn(dt)
        self.apply_pending_changes()


PROFILED_WORLD_METHODS = (
    "can_place_grass",
    "get_nearby_sheep",
    "get_nearby_wolves",
    "get_nearby_grass",
    "get_nearest_sheep",
    "get_nearest_wolf",
    "get_nearest_grass",
    "get_nearest_grass_entity",
    "count_nearby_grass",
    "_move_agents",
    "_resolve_physics_collisions",
    "_update_pending_asexual_births",
)

for _profiled_method_name in PROFILED_WORLD_METHODS:
    setattr(
        World,
        _profiled_method_name,
        profile_method(_profiled_method_name)(getattr(World, _profiled_method_name)),
    )


def main() -> None:
    if HEADLESS:
        world = World()
        profiler = FunctionProfiler(FPS)
        world._profiler = profiler
        recorder = PopulationRecorder(
            0.0, len(world.sheep_by_id), len(world.wolf_by_id), len(world.grass_by_id)
        )
        print("Headless mode running. Press Ctrl+C in this console to stop.")

        sim_time = 0.0
        sample_accum = 0.0
        target_dt = 1.0 / FPS
        previous_step_start = time.perf_counter() - target_dt
        current_fps = float(FPS)
        last_status_len = 0

        try:
            while True:
                step_start = time.perf_counter()
                loop_dt = step_start - previous_step_start
                previous_step_start = step_start
                if loop_dt > 1e-9:
                    current_fps = 1.0 / loop_dt

                world.step(target_dt)
                profiler.end_frame()
                sim_time += target_dt
                sample_accum += target_dt

                sheep_count = len(world.sheep_by_id)
                wolf_count = len(world.wolf_by_id)
                grass_count = len(world.grass_by_id)

                while sample_accum >= GRAPH_SAMPLE_INTERVAL_SEC:
                    sample_accum -= GRAPH_SAMPLE_INTERVAL_SEC
                    recorder.add_sample(
                        sim_time,
                        sheep_count,
                        wolf_count,
                        grass_count,
                    )

                status = (
                    f"FPS: {current_fps:6.1f} | "
                    f"Sheep: {sheep_count:4d} | "
                    f"Wolves: {wolf_count:4d} | "
                    f"Grass: {grass_count:4d}"
                )
                hotspots = ""
                if current_fps < 55.0:
                    hotspots = profiler.format_hotspots()
                if hotspots:
                    status = f"{status} | {hotspots}"
                padding = " " * max(0, last_status_len - len(status))
                print(f"\r{status}{padding}", end="", flush=True)
                last_status_len = len(status)

                remaining = target_dt - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print()

        if SAVE_TO_FILE:
            recorder.save_all()
        return

    runtime = {"recorder": PopulationRecorder(0.0, 0, 0, 0)}
    gui = SimulationGUI(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        sheep_scale=SHEEP_SCALE,
        wolf_scale=WOLF_SCALE,
        grass_scale=PLANT_SCALE,
        initial_sheep_count=0,
        initial_wolf_count=0,
        initial_grass_count=0,
        on_save_data=lambda: runtime["recorder"].save_all(),
        on_save_settings=save_simulation_settings,
        on_import_settings=load_simulation_settings,
        control_specs=SIMULATION_CONTROL_SPECS,
        toggle_specs=SIMULATION_TOGGLE_SPECS,
        control_values=default_simulation_settings(),
        plant_growth_sec=PLANT_GROWTH_SEC,
        animation_enabled=ANIMATION,
    )

    world: World | None = None
    sim_time = 0.0
    sample_accum = 0.0

    running = True
    while running:
        frame_dt = gui.tick()
        running = gui.handle_events()

        if gui.consume_clear_request():
            world = None
            runtime["recorder"] = PopulationRecorder(0.0, 0, 0, 0)
            sim_time = 0.0
            sample_accum = 0.0
            gui.set_simulation_loaded(False, paused=True)
            gui.clear_visuals()
            gui.reset_population_graphs(0, 0, 0)

        if gui.consume_start_request():
            apply_simulation_settings(gui.get_control_values())
            world = World()
            gui.set_painter_config(
                sheep_scale=SHEEP_SCALE,
                wolf_scale=WOLF_SCALE,
                grass_scale=PLANT_SCALE,
                plant_growth_sec=PLANT_GROWTH_SEC,
            )

            sheep_count = len(world.sheep_by_id)
            wolf_count = len(world.wolf_by_id)
            grass_count = len(world.grass_by_id)
            runtime["recorder"] = PopulationRecorder(
                0.0, sheep_count, wolf_count, grass_count
            )
            sim_time = 0.0
            sample_accum = 0.0
            gui.clear_visuals()
            gui.reset_population_graphs(sheep_count, wolf_count, grass_count)
            gui.set_simulation_loaded(True, paused=False)

        step_dt = 0.0
        if world is not None:
            step_dt = 0.0 if gui.paused else frame_dt
            if step_dt > 0.0:
                world.step(step_dt)

            sim_time += step_dt
            sample_accum += step_dt

            while sample_accum >= GRAPH_SAMPLE_INTERVAL_SEC:
                sample_accum -= GRAPH_SAMPLE_INTERVAL_SEC
                sheep_count = len(world.sheep_by_id)
                wolf_count = len(world.wolf_by_id)
                grass_count = len(world.grass_by_id)
                runtime["recorder"].add_sample(
                    sim_time, sheep_count, wolf_count, grass_count
                )
                gui.add_population_sample(
                    sim_time, sheep_count, wolf_count, grass_count
                )

        gui.update(frame_dt)
        gui.draw(world, sim_time, step_dt)

    gui.close()

    if SAVE_TO_FILE:
        runtime["recorder"].save_all()


if __name__ == "__main__":
    main()
