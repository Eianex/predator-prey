import math
import random
import time

from pygame.math import Vector2

from motors import MovementMotor, RandomWalkMotor, StraightLineMotor
from paint import SimulationGUI
from recorder import PopulationRecorder


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WIDTH, HEIGHT = 1000, 800
FPS = 60

ANIM_FRAME_COUNT = 120
ANIM_CYCLE_SEC = 0.5
ANIM_FPS = ANIM_FRAME_COUNT / ANIM_CYCLE_SEC

NUM_SHEEP = 40
NUM_WOLVES = 3
MAX_SHEEP = 100
INITIAL_PLANTS = 200

SHEEP_SCALE = 40
WOLF_SCALE = 50

SHEEP_SPEED = 100.0
WOLF_SPEED = 200.0

PLANT_SCALE = SHEEP_SCALE
PLANT_GROWTH_SEC = 5.0
PLANT_REPRODUCTION_PERIOD_SEC = 3.0
PLANT_REPRODUCTION_RADIUS = PLANT_SCALE
PLANT_NEARBY_RADIUS_MULT = 1.01
PLANT_NEARBY_LIMIT = 4
PLANT_RANDOM_SPAWN_CHANCE_PER_SEC = 0.001

SHEEP_STEP_SPEED_MULT_EXPAND = 0.8
SHEEP_STEP_SPEED_MULT_COMPRESS = 2 - SHEEP_STEP_SPEED_MULT_EXPAND
WOLF_STEP_SPEED_MULT_EXPAND = 0.8
WOLF_STEP_SPEED_MULT_COMPRESS = 2 - WOLF_STEP_SPEED_MULT_EXPAND
SHEEP_REPRODUCTION_COOLDOWN_SEC = 3.0
SHEEP_EAT_COOLDOWN_SEC = 4.0

GRAPH_SAMPLE_INTERVAL_SEC = 0.12
SAVE_TO_FILE = False
HEADLESS = False


# ------------------------------------------------------------
# Agents
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
        self.eat_cooldown = 0.0
        self.is_alive = True

        initial_angle = random.uniform(0.0, math.tau)
        self.vel = (
            Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.speed
        )
        self.base_radius = scale * 0.38
        self.motion_frame = random.uniform(0.0, float(ANIM_FRAME_COUNT))

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
        if self.eat_cooldown > 0.0:
            self.eat_cooldown = max(0.0, self.eat_cooldown - dt)

    def act(self, world: "World", dt: float) -> None:
        _ = dt
        self.try_eat_grass(world)
        self.try_reproduce(world)

    def try_eat_grass(self, world: "World") -> None:
        if self.id in world.pending_dead_ids:
            return
        if self.eat_cooldown > 0.0:
            return

        # Circle (sheep) against axis-aligned square (grass).
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
                return

    def try_reproduce(self, world: "World") -> None:
        if not self.can_reproduce() or self.id in world.pending_dead_ids:
            return

        search_radius = self.base_radius * 2.2
        nearby = world.get_nearby_sheep(self.pos, search_radius, exclude=self.id)
        for partner in nearby:
            if partner.id in world.pending_dead_ids:
                continue
            if partner.id <= self.id:
                continue
            min_dist = self.base_radius + partner.base_radius
            if (partner.pos - self.pos).length_squared() >= min_dist * min_dist:
                continue

            child = self.reproduce(partner, world)
            if child is not None:
                world.spawn_sheep(child)
            return

    def eat(self, target_plant: "Plant", world: "World") -> None:
        world.mark_dead(target_plant)
        self.eat_cooldown = SHEEP_EAT_COOLDOWN_SEC

    def die(self) -> None:
        self.is_alive = False

    def can_reproduce(self) -> bool:
        return self.is_alive and self.reproduction_cooldown <= 0.0

    def reproduce(self, partner: "Sheep", world: "World") -> "Sheep | None":
        if not self.can_reproduce() or not partner.can_reproduce():
            return None
        if not world.can_spawn_sheep():
            return None

        child_id = world.allocate_id()
        midpoint = (self.pos + partner.pos) * 0.5
        offset = Vector2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        if offset.length_squared() < 1e-6:
            offset = Vector2(1.0, 0.0)
        offset = offset.normalize() * random.uniform(0.0, self.base_radius)

        child = Sheep(
            animal_id=child_id,
            motor=RandomWalkMotor(),
            position=midpoint + offset,
            speed=SHEEP_SPEED,
            scale=SHEEP_SCALE,
            initial_reproduction_cooldown=SHEEP_REPRODUCTION_COOLDOWN_SEC,
        )
        self.reproduction_cooldown = SHEEP_REPRODUCTION_COOLDOWN_SEC
        partner.reproduction_cooldown = SHEEP_REPRODUCTION_COOLDOWN_SEC
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
        self.motion_frame = random.uniform(0.0, float(ANIM_FRAME_COUNT))

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
        _ = dt
        self.try_eat(world)

    def try_eat(self, world: "World") -> None:
        if self.id in world.pending_dead_ids:
            return

        search_radius = self.base_radius + (SHEEP_SCALE * 0.38)
        nearby = world.get_nearby_sheep(self.pos, search_radius)
        for sheep in nearby:
            if sheep.id in world.pending_dead_ids:
                continue
            min_dist = self.base_radius + sheep.base_radius
            if (sheep.pos - self.pos).length_squared() < min_dist * min_dist:
                self.eat(sheep, world)
                return

    def eat(self, target_sheep: Sheep, world: "World") -> None:
        world.mark_dead(target_sheep)

    def die(self) -> None:
        self.is_alive = False

    def reproduce(self) -> None:
        # Placeholder for future wolf reproduction behavior.
        return


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

        angle = random.uniform(0.0, math.tau)
        offset = Vector2(math.cos(angle), math.sin(angle)) * PLANT_REPRODUCTION_RADIUS
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
        self.pending_grass_births: list[Plant] = []
        self.pending_dead_ids: set[int] = set()

        for _ in range(NUM_SHEEP):
            sid = self.allocate_id()
            sheep = Sheep(
                animal_id=sid,
                motor=RandomWalkMotor(),
                position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                speed=SHEEP_SPEED,
                scale=SHEEP_SCALE,
            )
            self.sheep_by_id[sid] = sheep

        for _ in range(NUM_WOLVES):
            wid = self.allocate_id()
            wolf = Wolf(
                animal_id=wid,
                motor=StraightLineMotor(),
                position=Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                speed=WOLF_SPEED,
                scale=WOLF_SCALE,
            )
            self.wolf_by_id[wid] = wolf

        planted = 0
        attempts = 0
        max_attempts = INITIAL_PLANTS * 20
        while planted < INITIAL_PLANTS and attempts < max_attempts:
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

    def allocate_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

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
            sheep.pos, sheep.vel = self.bounce_in_bounds(
                sheep.pos, sheep.vel, sheep.base_radius, WIDTH, HEIGHT
            )
            self.pending_sheep_births.append(sheep)

    def spawn_grass(self, plant: Plant) -> None:
        if self.can_place_grass(plant):
            self.pending_grass_births.append(plant)

    def mark_dead(self, entity) -> None:
        self.pending_dead_ids.add(entity.id)

    def get_nearby_sheep(
        self, pos: Vector2, radius: float, exclude: int | None = None
    ) -> list[Sheep]:
        radius_sq = radius * radius
        nearby: list[Sheep] = []
        for sheep in self.sheep_by_id.values():
            if exclude is not None and sheep.id == exclude:
                continue
            if sheep.id in self.pending_dead_ids:
                continue
            if (sheep.pos - pos).length_squared() <= radius_sq:
                nearby.append(sheep)
        return nearby

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
        radius_sq = radius * radius
        nearby: list[Plant] = []
        for plant in self.grass_by_id.values():
            if exclude is not None and plant.id == exclude:
                continue
            if plant.id in self.pending_dead_ids:
                continue
            if (plant.pos - pos).length_squared() <= radius_sq:
                nearby.append(plant)
        return nearby

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
        if agent.vel.length_squared() < 1e-6:
            return
        agent.motion_frame = (agent.motion_frame + ANIM_FPS * dt) % ANIM_FRAME_COUNT

    def _move_agents(self, dt: float) -> None:
        for agent in self.all_agents():
            scale = self._displacement_scale(agent)
            agent.move(dt, scale)
            agent.pos, agent.vel = self.bounce_in_bounds(
                agent.pos, agent.vel, agent.base_radius, WIDTH, HEIGHT
            )
            self._advance_motion_frame(agent, dt)

    def _resolve_physics_collisions(self) -> None:
        agents = self.all_agents()
        for i in range(len(agents)):
            a = agents[i]
            if a.id in self.pending_dead_ids:
                continue
            for j in range(i + 1, len(agents)):
                b = agents[j]
                if b.id in self.pending_dead_ids:
                    continue

                if (isinstance(a, Sheep) and isinstance(b, Wolf)) or (
                    isinstance(a, Wolf) and isinstance(b, Sheep)
                ):
                    continue

                delta = b.pos - a.pos
                min_dist = a.base_radius + b.base_radius
                dist_sq = delta.length_squared()
                if dist_sq >= min_dist * min_dist:
                    continue

                if dist_sq < 1e-12:
                    angle = random.uniform(0.0, math.tau)
                    normal = Vector2(math.cos(angle), math.sin(angle))
                    dist = 0.0
                else:
                    dist = math.sqrt(dist_sq)
                    normal = delta / dist

                self.elastic_collision_response(a, b, normal, dist, min_dist)

    def _act_agents(self, dt: float) -> None:
        for wolf in list(self.wolf_by_id.values()):
            if wolf.id in self.pending_dead_ids:
                continue
            wolf.act(self, dt)

        for sheep in list(self.sheep_by_id.values()):
            if sheep.id in self.pending_dead_ids:
                continue
            sheep.act(self, dt)

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
                sheep.die()
            wolf = self.wolf_by_id.pop(dead_id, None)
            if wolf is not None:
                wolf.die()
            plant = self.grass_by_id.pop(dead_id, None)
            if plant is not None:
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

        if self.pending_grass_births:
            for child in self.pending_grass_births:
                if child.id in self.grass_by_id:
                    continue
                if self.can_place_grass(child, exclude=child.id):
                    self.grass_by_id[child.id] = child
            self.pending_grass_births.clear()

    def step(self, dt: float) -> None:
        self._move_agents(dt)
        self._resolve_physics_collisions()
        self._act_agents(dt)
        self._update_grass(dt)
        self._try_random_grass_spawn(dt)
        self.apply_pending_changes()


def main() -> None:
    world = World()
    recorder = PopulationRecorder(
        0.0, len(world.sheep_by_id), len(world.wolf_by_id), len(world.grass_by_id)
    )
    if HEADLESS:
        print("Headless mode running. Press Ctrl+C in this console to stop.")

        sim_time = 0.0
        sample_accum = 0.0
        last_time = time.perf_counter()

        try:
            while True:
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                if dt <= 0.0:
                    dt = 1.0 / FPS

                world.step(dt)
                sim_time += dt
                sample_accum += dt

                while sample_accum >= GRAPH_SAMPLE_INTERVAL_SEC:
                    sample_accum -= GRAPH_SAMPLE_INTERVAL_SEC
                    recorder.add_sample(
                        sim_time,
                        len(world.sheep_by_id),
                        len(world.wolf_by_id),
                        len(world.grass_by_id),
                    )

                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
    else:
        gui = SimulationGUI(
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            sheep_scale=SHEEP_SCALE,
            wolf_scale=WOLF_SCALE,
            initial_sheep_count=len(world.sheep_by_id),
            initial_wolf_count=len(world.wolf_by_id),
            initial_grass_count=len(world.grass_by_id),
            on_save_sheep=recorder.save_sheep,
            on_save_wolf=recorder.save_wolf,
            on_save_grass=recorder.save_grass,
        )

        sim_time = 0.0
        sample_accum = 0.0

        running = True
        while running:
            frame_dt = gui.tick()
            running = gui.handle_events()

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
                recorder.add_sample(sim_time, sheep_count, wolf_count, grass_count)
                gui.add_population_sample(
                    sim_time, sheep_count, wolf_count, grass_count
                )

            gui.update(frame_dt)
            gui.draw(world, sim_time, step_dt)

        gui.close()

    if SAVE_TO_FILE:
        recorder.save_all()


if __name__ == "__main__":
    main()
