import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pygame
from pygame.math import Vector2


PANEL_WIDTH = 400
PANEL_BG_COLOR = (22, 28, 30)
PANEL_BORDER_COLOR = (70, 86, 90)
GRAPH_BG_COLOR = (16, 21, 24)
SHEEP_GRAPH_COLOR = (188, 246, 166)
WOLF_GRAPH_COLOR = (246, 148, 120)
WORLD_BG_COLOR = (78, 145, 68)

ASSET_DIR = Path("img")
ANIM_DIR = ASSET_DIR / "animation"
SHEEP_ANIM_DIR = ANIM_DIR / "sheep"
WOLF_ANIM_DIR = ANIM_DIR / "wolf"
PLANT_ANIM_DIR = ANIM_DIR / "plant"
TURN_DURATION_SEC = 0.5
PLANT_GROWTH_SEC = 5.0
SHOW_GRAPHS = True


@dataclass
class AnimalVisual:
    display_angle: float
    target_angle: float
    turn_start_angle: float
    turn_elapsed: float


class Painter:
    def __init__(
        self,
        sheep_scale: int,
        wolf_scale: int,
    ):
        self.turn_duration_sec = TURN_DURATION_SEC
        self.sheep_animation_frames = Painter.load_animation_frames(
            SHEEP_ANIM_DIR,
            "sheep",
            sheep_scale,
        )
        self.wolf_animation_frames = Painter.load_animation_frames(
            WOLF_ANIM_DIR,
            "wolf",
            wolf_scale,
        )
        self.plant_animation_frames = Painter.load_animation_frames(
            PLANT_ANIM_DIR,
            "plant",
            sheep_scale,
        )
        self.visuals_by_id: dict[int, AnimalVisual] = {}

    @staticmethod
    def load_image(path: Path, size: int) -> pygame.Surface:
        surface = pygame.image.load(path.as_posix()).convert_alpha()
        return pygame.transform.smoothscale(surface, (size, size))

    @staticmethod
    def load_animation_frames(
        directory: Path, prefix: str, size: int
    ) -> list[pygame.Surface]:
        frames: list[pygame.Surface] = []
        frame_paths = sorted(directory.glob(f"{prefix}[0-9][0-9][0-9][0-9].png"))
        for frame_path in frame_paths:
            frames.append(Painter.load_image(frame_path, size))
        if len(frames) == 0:
            raise FileNotFoundError(
                f"No animation frames found in {directory} for prefix '{prefix}'"
            )
        return frames

    @staticmethod
    def _velocity_to_display_angle(vel: Vector2) -> float:
        return math.degrees(math.atan2(-vel.x, vel.y))

    @staticmethod
    def angle_diff_deg(current: float, target: float) -> float:
        return (target - current + 180.0) % 360.0 - 180.0

    def _ensure_visual(self, agent) -> AnimalVisual:
        visual = self.visuals_by_id.get(agent.id)
        if visual is not None:
            return visual

        initial = Painter._velocity_to_display_angle(agent.vel)
        visual = AnimalVisual(
            display_angle=initial,
            target_angle=initial,
            turn_start_angle=initial,
            turn_elapsed=self.turn_duration_sec,
        )
        self.visuals_by_id[agent.id] = visual
        return visual

    def _update_visual(self, agent, dt: float) -> AnimalVisual:
        visual = self._ensure_visual(agent)

        if agent.vel.length_squared() > 1e-6:
            new_target = Painter._velocity_to_display_angle(agent.vel)
            if abs(Painter.angle_diff_deg(visual.target_angle, new_target)) > 1e-3:
                visual.turn_start_angle = visual.display_angle
                visual.target_angle = new_target
                visual.turn_elapsed = 0.0

        if self.turn_duration_sec <= 1e-6:
            visual.display_angle = visual.target_angle
            visual.turn_elapsed = self.turn_duration_sec
        elif visual.turn_elapsed < self.turn_duration_sec:
            visual.turn_elapsed = min(self.turn_duration_sec, visual.turn_elapsed + dt)
            progress = visual.turn_elapsed / self.turn_duration_sec
            delta = Painter.angle_diff_deg(visual.turn_start_angle, visual.target_angle)
            visual.display_angle = visual.turn_start_angle + delta * progress
        else:
            visual.display_angle = visual.target_angle

        return visual

    def _draw_agent(
        self,
        screen: pygame.Surface,
        agent,
        frames: list[pygame.Surface],
        dt: float,
        x_offset: int,
    ) -> None:
        visual = self._update_visual(agent, dt)
        if len(frames) == 0:
            return

        frame_cursor = agent.motion_frame
        frame_index = int(frame_cursor) % len(frames)
        base_image = frames[frame_index]

        render_angle = -visual.display_angle
        rotated = pygame.transform.rotozoom(base_image, render_angle, 1.0)
        rect = rotated.get_rect(center=(agent.pos.x + x_offset, agent.pos.y))
        screen.blit(rotated, rect)

        heading_angle_rad = math.radians(render_angle)
        direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        line_start = Vector2(agent.pos.x + x_offset, agent.pos.y)
        line_end = line_start + direction * (agent.base_radius + 18)
        pygame.draw.line(
            screen,
            (245, 245, 245),
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            2,
        )

    def _draw_plant(
        self,
        screen: pygame.Surface,
        plant,
        x_offset: int,
    ) -> None:
        if len(self.plant_animation_frames) == 0:
            return

        growth_ratio = max(0.0, min(1.0, plant.age_sec / PLANT_GROWTH_SEC))
        frame_index = int(growth_ratio * (len(self.plant_animation_frames) - 1))
        image = self.plant_animation_frames[frame_index]
        rect = image.get_rect(center=(plant.pos.x + x_offset, plant.pos.y))
        screen.blit(image, rect)

    def draw(self, screen: pygame.Surface, world, dt: float, x_offset: int = 0) -> None:
        live_ids = world.live_ids()
        stale_ids = [aid for aid in self.visuals_by_id.keys() if aid not in live_ids]
        for aid in stale_ids:
            self.visuals_by_id.pop(aid, None)

        for plant in world.grass_by_id.values():
            self._draw_plant(screen, plant, x_offset)

        for sheep in world.sheep_by_id.values():
            self._draw_agent(screen, sheep, self.sheep_animation_frames, dt, x_offset)
        for wolf in world.wolf_by_id.values():
            self._draw_agent(screen, wolf, self.wolf_animation_frames, dt, x_offset)


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
        self.save_button_rect = pygame.Rect(
            self.rect.right - 96, self.rect.y + 8, 84, 22
        )
        self.save_flash_timer = 0.0

    def add_sample(self, time_sec: float, value: int) -> None:
        self.samples.append((time_sec, float(value)))

    def mark_saved(self) -> None:
        self.save_flash_timer = 1.0

    def is_save_button_clicked(self, mouse_pos: tuple[int, int]) -> bool:
        return self.save_button_rect.collidepoint(mouse_pos)

    def update(self, dt: float) -> None:
        target_value = self.samples[-1][1]
        alpha = min(1.0, dt * 6.0)
        self.display_latest += (target_value - self.display_latest) * alpha

        visible = [v for (_, v) in self.samples]
        if not visible:
            visible = [target_value]
        target_max = max(5.0, max(max(visible), self.display_latest) * 1.15)
        max_alpha = min(1.0, dt * 3.0)
        self.display_y_max += (target_max - self.display_y_max) * max_alpha

        if self.save_flash_timer > 0.0:
            self.save_flash_timer = max(0.0, self.save_flash_timer - dt)

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

        if self.save_flash_timer > 0.0:
            btn_color = (66, 120, 70)
            btn_label = "Saved"
        else:
            btn_color = (47, 62, 66)
            btn_label = "Save CSV"
        pygame.draw.rect(surface, btn_color, self.save_button_rect, border_radius=6)
        pygame.draw.rect(
            surface,
            (120, 140, 145),
            self.save_button_rect,
            width=1,
            border_radius=6,
        )
        btn_text = small_font.render(btn_label, True, (225, 236, 230))
        btn_rect = btn_text.get_rect(center=self.save_button_rect.center)
        surface.blit(btn_text, btn_rect)

        plot_rect = pygame.Rect(
            self.rect.x + 12,
            self.rect.y + 56,
            self.rect.width - 24,
            self.rect.height - 72,
        )

        for i in range(5):
            gy = plot_rect.y + int(i * (plot_rect.height / 4))
            pygame.draw.line(
                surface, (45, 58, 61), (plot_rect.x, gy), (plot_rect.right, gy), 1
            )
        pygame.draw.rect(surface, (70, 86, 90), plot_rect, width=1)

        visible = self.samples
        if not visible:
            visible = [(current_time_sec, self.samples[-1][1])]

        y_max = max(1.0, self.display_y_max)
        total_time = max(current_time_sec, visible[-1][0], 1e-6)

        def to_screen(t: float, v: float) -> tuple[int, int]:
            tx = t / total_time
            tx = max(0.0, min(1.0, tx))
            ty = max(0.0, min(1.0, v / y_max))
            x = plot_rect.x + int(tx * plot_rect.width)
            y = plot_rect.bottom - int(ty * plot_rect.height)
            return x, y

        if len(visible) >= 2:
            pts = [to_screen(t, v) for (t, v) in visible]
            pygame.draw.lines(surface, self.color, False, pts, 2)

        dot_x = plot_rect.right
        dot_y = plot_rect.bottom - int(
            max(0.0, min(1.0, self.display_latest / y_max)) * plot_rect.height
        )
        pygame.draw.circle(surface, self.color, (dot_x, dot_y), 5)

        max_label = small_font.render(f"{int(round(y_max))}", True, (150, 165, 168))
        zero_label = small_font.render("0", True, (150, 165, 168))
        surface.blit(max_label, (plot_rect.x + 4, plot_rect.y + 2))
        surface.blit(zero_label, (plot_rect.x + 4, plot_rect.bottom - 16))


class SimulationGUI:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        sheep_scale: int,
        wolf_scale: int,
        initial_sheep_count: int,
        initial_wolf_count: int,
        on_save_sheep: Callable[[], None],
        on_save_wolf: Callable[[], None],
    ):
        pygame.init()
        pygame.display.set_caption("Wolf-Sheep Prototype")

        self.width = width
        self.height = height
        self.fps = fps
        self.show_graphs = SHOW_GRAPHS
        self.total_width = width + PANEL_WIDTH if self.show_graphs else width
        self.world_x_offset = PANEL_WIDTH if self.show_graphs else 0

        self.screen = pygame.display.set_mode((self.total_width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)

        self.panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, self.height)
        self.world_rect = pygame.Rect(self.world_x_offset, 0, self.width, self.height)
        self.pause_button_rect = pygame.Rect(self.total_width - 118, 10, 108, 30)
        self.paused = False

        self.on_save_sheep = on_save_sheep
        self.on_save_wolf = on_save_wolf

        self.painter = Painter(
            sheep_scale,
            wolf_scale,
        )

        self.sheep_graph: PopulationGraph | None = None
        self.wolf_graph: PopulationGraph | None = None
        if self.show_graphs:
            margin = 14
            gap = 12
            graph_height = (self.height - margin * 2 - gap) // 2
            self.sheep_graph = PopulationGraph(
                pygame.Rect(margin, margin, PANEL_WIDTH - margin * 2, graph_height),
                "Sheep Population",
                SHEEP_GRAPH_COLOR,
                initial_sheep_count,
            )
            self.wolf_graph = PopulationGraph(
                pygame.Rect(
                    margin,
                    margin + graph_height + gap,
                    PANEL_WIDTH - margin * 2,
                    graph_height,
                ),
                "Wolf Population",
                WOLF_GRAPH_COLOR,
                initial_wolf_count,
            )

    def tick(self) -> float:
        return self.clock.tick(self.fps) / 1000.0

    def handle_events(self) -> bool:
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.pause_button_rect.collidepoint(event.pos):
                    self.paused = not self.paused
                elif self.show_graphs:
                    if (
                        self.sheep_graph is not None
                        and self.sheep_graph.is_save_button_clicked(event.pos)
                    ):
                        self.on_save_sheep()
                        self.sheep_graph.mark_saved()
                    elif (
                        self.wolf_graph is not None
                        and self.wolf_graph.is_save_button_clicked(event.pos)
                    ):
                        self.on_save_wolf()
                        self.wolf_graph.mark_saved()
        return running

    def add_population_sample(
        self, time_sec: float, sheep_count: int, wolf_count: int
    ) -> None:
        if self.sheep_graph is not None:
            self.sheep_graph.add_sample(time_sec, sheep_count)
        if self.wolf_graph is not None:
            self.wolf_graph.add_sample(time_sec, wolf_count)

    def update(self, frame_dt: float) -> None:
        if self.sheep_graph is not None:
            self.sheep_graph.update(frame_dt)
        if self.wolf_graph is not None:
            self.wolf_graph.update(frame_dt)

    def draw(self, world, sim_time: float, step_dt: float) -> None:
        self.screen.fill((0, 0, 0))
        if self.show_graphs:
            pygame.draw.rect(self.screen, PANEL_BG_COLOR, self.panel_rect)
            pygame.draw.rect(self.screen, WORLD_BG_COLOR, self.world_rect)
            pygame.draw.line(
                self.screen,
                PANEL_BORDER_COLOR,
                (PANEL_WIDTH, 0),
                (PANEL_WIDTH, self.height),
                2,
            )
            if self.sheep_graph is not None:
                self.sheep_graph.draw(self.screen, self.font, self.small_font, sim_time)
            if self.wolf_graph is not None:
                self.wolf_graph.draw(self.screen, self.font, self.small_font, sim_time)
        else:
            pygame.draw.rect(self.screen, WORLD_BG_COLOR, self.world_rect)

        self.painter.draw(self.screen, world, step_dt, x_offset=self.world_x_offset)

        fps_x = 12 if self.show_graphs else self.world_x_offset + 12
        fps_text = self.small_font.render(
            f"FPS: {self.clock.get_fps():5.1f}", True, (220, 230, 225)
        )
        self.screen.blit(fps_text, (fps_x, self.height - 24))

        pause_label = "Continue" if self.paused else "Pause"
        pause_bg = (142, 88, 76) if self.paused else (74, 108, 80)
        pygame.draw.rect(self.screen, pause_bg, self.pause_button_rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            (220, 230, 225),
            self.pause_button_rect,
            width=1,
            border_radius=8,
        )
        pause_text = self.small_font.render(pause_label, True, (245, 245, 245))
        pause_text_rect = pause_text.get_rect(center=self.pause_button_rect.center)
        self.screen.blit(pause_text, pause_text_rect)

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
