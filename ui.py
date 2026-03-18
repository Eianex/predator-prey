import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pygame
from pygame.math import Vector2


CONTROL_PANEL_WIDTH = 350
GRAPH_PANEL_WIDTH = 350
PANEL_BG_COLOR = (22, 28, 30)
PANEL_BORDER_COLOR = (70, 86, 90)
GRAPH_BG_COLOR = (16, 21, 24)
CONTROL_BG_COLOR = (18, 22, 24)
CONTROL_INPUT_BG = (31, 40, 43)
CONTROL_INPUT_ACTIVE_BG = (48, 62, 66)
CONTROL_TRACK_COLOR = (54, 66, 71)
CONTROL_FILL_COLOR = (124, 190, 150)
CONTROL_KNOB_COLOR = (226, 236, 232)
BUTTON_BG_COLOR = (47, 62, 66)
BUTTON_BORDER_COLOR = (120, 140, 145)
BUTTON_PLAY_COLOR = (74, 108, 80)
BUTTON_PAUSE_COLOR = (142, 88, 76)
BUTTON_CLEAR_COLOR = (112, 82, 70)
SHEEP_GRAPH_COLOR = (188, 246, 166)
WOLF_GRAPH_COLOR = (246, 148, 120)
GRASS_GRAPH_COLOR = (150, 219, 95)
WORLD_BG_COLOR = (78, 145, 68)

ASSET_DIR = Path("img")
ANIM_DIR = ASSET_DIR / "animation"
SHEEP_ANIM_DIR = ANIM_DIR / "sheep"
WOLF_ANIM_DIR = ANIM_DIR / "wolf"
PLANT_ANIM_DIR = ANIM_DIR / "plant"
TURN_DURATION_SEC = 0.5
DEFAULT_PLANT_GROWTH_SEC = 5.0
SHOW_GRAPHS = True


@dataclass
class AnimalVisual:
    display_angle: float
    target_angle: float
    turn_start_angle: float
    turn_elapsed: float


@dataclass
class ControlSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    integer: bool = False
    decimals: int = 2


@dataclass
class ToggleSpec:
    key: str
    label: str


class ToggleControl:
    def __init__(self, spec: ToggleSpec, value: bool, rect: pygame.Rect):
        self.spec = spec
        self.rect = rect
        self.value = bool(value)
        self.checkbox_rect = pygame.Rect(rect.right - 36, rect.y + 7, 20, 20)

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def toggle(self) -> None:
        self.value = not self.value

    def draw(
        self,
        surface: pygame.Surface,
        label_font: pygame.font.Font,
    ) -> None:
        pygame.draw.rect(surface, CONTROL_BG_COLOR, self.rect, border_radius=8)
        pygame.draw.rect(
            surface, PANEL_BORDER_COLOR, self.rect, width=1, border_radius=8
        )

        label = label_font.render(self.spec.label, True, (224, 234, 230))
        label_rect = label.get_rect(midleft=(self.rect.x + 12, self.rect.centery))
        surface.blit(label, label_rect)

        checkbox_bg = CONTROL_FILL_COLOR if self.value else CONTROL_INPUT_BG
        pygame.draw.rect(surface, checkbox_bg, self.checkbox_rect, border_radius=5)
        pygame.draw.rect(
            surface,
            BUTTON_BORDER_COLOR,
            self.checkbox_rect,
            width=1,
            border_radius=5,
        )
        if self.value:
            start = (self.checkbox_rect.x + 4, self.checkbox_rect.centery)
            mid = (self.checkbox_rect.x + 8, self.checkbox_rect.bottom - 5)
            end = (self.checkbox_rect.right - 4, self.checkbox_rect.y + 5)
            pygame.draw.lines(surface, (24, 35, 28), False, [start, mid, end], 3)


class NumericControl:
    def __init__(self, spec: ControlSpec, value: float, rect: pygame.Rect):
        self.spec = spec
        self.rect = rect
        self.value = 0.0
        self.dragging = False
        self.active_text = False
        self.input_text = ""

        self.slider_rect = pygame.Rect(rect.x + 12, rect.y + 23, rect.width - 110, 6)
        self.input_rect = pygame.Rect(rect.right - 84, rect.y + 7, 72, 20)
        self.set_value(value)

    def _quantize(self, value: float) -> float:
        value = max(self.spec.minimum, min(self.spec.maximum, value))
        if self.spec.step > 0.0:
            steps = round((value - self.spec.minimum) / self.spec.step)
            value = self.spec.minimum + steps * self.spec.step
        value = max(self.spec.minimum, min(self.spec.maximum, value))
        if self.spec.integer:
            return float(int(round(value)))
        return float(value)

    def _format_value(self, value: float) -> str:
        if self.spec.integer:
            return str(int(round(value)))
        return f"{value:.{self.spec.decimals}f}"

    def set_value(self, value: float) -> None:
        self.value = self._quantize(value)
        if not self.active_text:
            self.input_text = self._format_value(self.value)

    def contains_slider(self, pos: tuple[int, int]) -> bool:
        return self.slider_rect.inflate(0, 16).collidepoint(pos)

    def contains_input(self, pos: tuple[int, int]) -> bool:
        return self.input_rect.collidepoint(pos)

    def begin_drag(self, pos: tuple[int, int]) -> None:
        self.dragging = True
        self.active_text = False
        self._set_from_pos(pos[0])

    def update_drag(self, pos: tuple[int, int]) -> None:
        if self.dragging:
            self._set_from_pos(pos[0])

    def end_drag(self) -> None:
        self.dragging = False

    def _set_from_pos(self, x: int) -> None:
        if self.slider_rect.width <= 0:
            return
        ratio = (x - self.slider_rect.x) / self.slider_rect.width
        ratio = max(0.0, min(1.0, ratio))
        raw_value = self.spec.minimum + ratio * (self.spec.maximum - self.spec.minimum)
        self.set_value(raw_value)

    def activate_text(self) -> None:
        self.active_text = True
        self.input_text = self._format_value(self.value)

    def deactivate_text(self, commit: bool) -> None:
        if self.active_text and commit:
            self.commit_text()
        self.active_text = False
        self.input_text = self._format_value(self.value)

    def commit_text(self) -> None:
        candidate = self.input_text.strip().replace(",", ".")
        if candidate in {"", ".", "-", "-."}:
            self.input_text = self._format_value(self.value)
            return
        try:
            parsed = float(candidate)
        except ValueError:
            self.input_text = self._format_value(self.value)
            return
        self.set_value(parsed)

    def handle_text_input(self, text: str) -> None:
        if not self.active_text:
            return
        allowed = "0123456789.-"
        filtered = "".join(ch for ch in text if ch in allowed)
        if filtered:
            self.input_text += filtered

    def handle_keydown(self, event: pygame.event.Event) -> None:
        if not self.active_text:
            return
        if event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.deactivate_text(True)
        elif event.key == pygame.K_ESCAPE:
            self.deactivate_text(False)

    def draw(
        self,
        surface: pygame.Surface,
        label_font: pygame.font.Font,
        input_font: pygame.font.Font,
    ) -> None:
        pygame.draw.rect(surface, CONTROL_BG_COLOR, self.rect, border_radius=8)
        pygame.draw.rect(
            surface, PANEL_BORDER_COLOR, self.rect, width=1, border_radius=8
        )

        label = label_font.render(self.spec.label, True, (224, 234, 230))
        surface.blit(label, (self.rect.x + 12, self.rect.y + 6))

        pygame.draw.line(
            surface,
            CONTROL_TRACK_COLOR,
            (self.slider_rect.x, self.slider_rect.centery),
            (self.slider_rect.right, self.slider_rect.centery),
            4,
        )

        if self.spec.maximum - self.spec.minimum <= 1e-9:
            ratio = 0.0
        else:
            ratio = (self.value - self.spec.minimum) / (
                self.spec.maximum - self.spec.minimum
            )
        ratio = max(0.0, min(1.0, ratio))
        knob_x = self.slider_rect.x + int(ratio * self.slider_rect.width)
        pygame.draw.line(
            surface,
            CONTROL_FILL_COLOR,
            (self.slider_rect.x, self.slider_rect.centery),
            (knob_x, self.slider_rect.centery),
            4,
        )
        pygame.draw.circle(
            surface, CONTROL_KNOB_COLOR, (knob_x, self.slider_rect.centery), 7
        )
        pygame.draw.circle(
            surface, (35, 43, 46), (knob_x, self.slider_rect.centery), 7, 1
        )

        input_bg = CONTROL_INPUT_ACTIVE_BG if self.active_text else CONTROL_INPUT_BG
        pygame.draw.rect(surface, input_bg, self.input_rect, border_radius=6)
        pygame.draw.rect(
            surface, BUTTON_BORDER_COLOR, self.input_rect, width=1, border_radius=6
        )
        text_value = (
            self.input_text if self.active_text else self._format_value(self.value)
        )
        text_surface = input_font.render(text_value, True, (240, 245, 242))
        text_rect = text_surface.get_rect(center=self.input_rect.center)
        surface.blit(text_surface, text_rect)


class Painter:
    def __init__(
        self,
        sheep_scale: int,
        wolf_scale: int,
        grass_scale: int,
        plant_growth_sec: float,
        animation_enabled: bool,
    ):
        self.turn_duration_sec = TURN_DURATION_SEC
        self.plant_growth_sec = max(1e-6, plant_growth_sec)
        self.animation_enabled = animation_enabled
        self.sheep_scale = sheep_scale
        self.wolf_scale = wolf_scale
        self.grass_scale = grass_scale
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
            grass_scale,
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

    def clear(self) -> None:
        self.visuals_by_id.clear()

    def set_config(
        self,
        sheep_scale: int,
        wolf_scale: int,
        grass_scale: int,
        plant_growth_sec: float,
    ) -> None:
        self.plant_growth_sec = max(1e-6, plant_growth_sec)
        self.sheep_scale = sheep_scale
        self.wolf_scale = wolf_scale
        self.grass_scale = grass_scale
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
            grass_scale,
        )
        self.clear()

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

        if self.animation_enabled:
            frame_cursor = agent.motion_frame
            frame_index = int(frame_cursor) % len(frames)
        else:
            frame_index = 0
        base_image = frames[frame_index]

        render_angle = -visual.display_angle
        rotated = pygame.transform.rotozoom(base_image, render_angle, 1.0)
        rect = rotated.get_rect(center=(agent.pos.x + x_offset, agent.pos.y))
        screen.blit(rotated, rect)

        # heading_angle_rad = math.radians(render_angle)
        # direction = Vector2(math.sin(heading_angle_rad), math.cos(heading_angle_rad))
        # line_start = Vector2(agent.pos.x + x_offset, agent.pos.y)
        # line_end = line_start + direction * (agent.base_radius + 18)
        # pygame.draw.line(
        # screen,
        # (245, 245, 245),
        # (line_start.x, line_start.y),
        # (line_end.x, line_end.y),
        # 2,
        # )

    def _draw_plant(
        self,
        screen: pygame.Surface,
        plant,
        x_offset: int,
    ) -> None:
        if len(self.plant_animation_frames) == 0:
            return

        growth_ratio = max(0.0, min(1.0, plant.age_sec / self.plant_growth_sec))
        frame_index = int(growth_ratio * (len(self.plant_animation_frames) - 1))
        image = self.plant_animation_frames[frame_index]
        rect = image.get_rect(center=(plant.pos.x + x_offset, plant.pos.y))
        screen.blit(image, rect)

    def draw(self, screen: pygame.Surface, world, dt: float, x_offset: int = 0) -> None:
        if world is None:
            return

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
        self.save_button_rect = pygame.Rect(
            self.rect.right - 96, self.rect.y + 8, 84, 22
        )
        self.save_flash_timer = 0.0
        self.reset(initial_value)

    def reset(self, initial_value: int) -> None:
        self.samples: list[tuple[float, float]] = [(0.0, float(initial_value))]
        self.display_latest = float(initial_value)
        self.display_y_max = max(10.0, float(initial_value) + 3.0)
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
        grass_scale: int,
        initial_sheep_count: int,
        initial_wolf_count: int,
        initial_grass_count: int,
        on_save_sheep: Callable[[], None],
        on_save_wolf: Callable[[], None],
        on_save_grass: Callable[[], None],
        control_specs: list[dict],
        toggle_specs: list[dict],
        control_values: dict[str, float | bool],
        plant_growth_sec: float = DEFAULT_PLANT_GROWTH_SEC,
        animation_enabled: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption("Predator Prey Simulation")

        self.width = width
        self.height = height
        self.fps = fps
        self.show_graphs = SHOW_GRAPHS
        self.total_width = (
            width + CONTROL_PANEL_WIDTH + (GRAPH_PANEL_WIDTH if self.show_graphs else 0)
        )
        self.world_x_offset = CONTROL_PANEL_WIDTH

        self.screen = pygame.display.set_mode((self.total_width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.tiny_font = pygame.font.SysFont("consolas", 13)

        self.control_panel_rect = pygame.Rect(0, 0, CONTROL_PANEL_WIDTH, self.height)
        self.world_rect = pygame.Rect(self.world_x_offset, 0, self.width, self.height)
        self.graph_panel_rect = pygame.Rect(
            self.world_rect.right,
            0,
            GRAPH_PANEL_WIDTH,
            self.height,
        )

        button_y = 44
        button_width = (CONTROL_PANEL_WIDTH - 44) // 2
        self.play_button_rect = pygame.Rect(14, button_y, button_width, 34)
        self.clear_button_rect = pygame.Rect(
            28 + button_width, button_y, button_width, 34
        )

        self.paused = True
        self.simulation_loaded = False
        self._pending_start_request = False
        self._pending_clear_request = False

        self.on_save_sheep = on_save_sheep
        self.on_save_wolf = on_save_wolf
        self.on_save_grass = on_save_grass

        self.painter = Painter(
            sheep_scale,
            wolf_scale,
            grass_scale,
            plant_growth_sec=plant_growth_sec,
            animation_enabled=animation_enabled,
        )

        self.controls = self._build_controls(control_specs, control_values)
        toggle_start_y = 108 + len(self.controls) * 37 + 10
        self.toggle_controls = self._build_toggle_controls(
            toggle_specs,
            control_values,
            toggle_start_y,
        )
        self.sheep_graph: PopulationGraph | None = None
        self.wolf_graph: PopulationGraph | None = None
        self.grass_graph: PopulationGraph | None = None
        self._build_graphs(initial_sheep_count, initial_wolf_count, initial_grass_count)

    def _build_controls(
        self, control_specs: list[dict], control_values: dict[str, float | bool]
    ) -> list[NumericControl]:
        controls: list[NumericControl] = []
        start_y = 108
        row_height = 34
        gap = 3
        rect = pygame.Rect(12, start_y, CONTROL_PANEL_WIDTH - 24, row_height)
        for spec_data in control_specs:
            spec = ControlSpec(**spec_data)
            value = float(control_values.get(spec.key, spec.minimum))
            controls.append(NumericControl(spec, value, rect.copy()))
            rect.y += row_height + gap
        return controls

    def _build_toggle_controls(
        self,
        toggle_specs: list[dict],
        control_values: dict[str, float | bool],
        start_y: int,
    ) -> list[ToggleControl]:
        controls: list[ToggleControl] = []
        row_height = 34
        gap = 6
        rect = pygame.Rect(12, start_y, CONTROL_PANEL_WIDTH - 24, row_height)
        for spec_data in toggle_specs:
            spec = ToggleSpec(**spec_data)
            value = bool(control_values.get(spec.key, False))
            controls.append(ToggleControl(spec, value, rect.copy()))
            rect.y += row_height + gap
        return controls

    def _build_graphs(
        self,
        initial_sheep_count: int,
        initial_wolf_count: int,
        initial_grass_count: int,
    ) -> None:
        if not self.show_graphs:
            self.sheep_graph = None
            self.wolf_graph = None
            self.grass_graph = None
            return
        margin = 14
        gap = 12
        graph_height = (self.height - margin * 2 - gap * 2) // 3
        base_x = self.graph_panel_rect.x + margin
        self.sheep_graph = PopulationGraph(
            pygame.Rect(base_x, margin, GRAPH_PANEL_WIDTH - margin * 2, graph_height),
            "Sheep Population",
            SHEEP_GRAPH_COLOR,
            initial_sheep_count,
        )
        self.wolf_graph = PopulationGraph(
            pygame.Rect(
                base_x,
                margin + graph_height + gap,
                GRAPH_PANEL_WIDTH - margin * 2,
                graph_height,
            ),
            "Wolf Population",
            WOLF_GRAPH_COLOR,
            initial_wolf_count,
        )
        self.grass_graph = PopulationGraph(
            pygame.Rect(
                base_x,
                margin + (graph_height + gap) * 2,
                GRAPH_PANEL_WIDTH - margin * 2,
                graph_height,
            ),
            "Grass Population",
            GRASS_GRAPH_COLOR,
            initial_grass_count,
        )

    def reset_population_graphs(
        self, sheep_count: int, wolf_count: int, grass_count: int
    ) -> None:
        if self.sheep_graph is not None:
            self.sheep_graph.reset(sheep_count)
        if self.wolf_graph is not None:
            self.wolf_graph.reset(wolf_count)
        if self.grass_graph is not None:
            self.grass_graph.reset(grass_count)

    def clear_visuals(self) -> None:
        self.painter.clear()

    def set_painter_config(
        self,
        sheep_scale: int,
        wolf_scale: int,
        grass_scale: int,
        plant_growth_sec: float,
    ) -> None:
        self.painter.set_config(
            sheep_scale=sheep_scale,
            wolf_scale=wolf_scale,
            grass_scale=grass_scale,
            plant_growth_sec=plant_growth_sec,
        )

    def set_simulation_loaded(self, loaded: bool, paused: bool) -> None:
        self.simulation_loaded = loaded
        self.paused = paused

    def consume_start_request(self) -> bool:
        pending = self._pending_start_request
        self._pending_start_request = False
        return pending

    def consume_clear_request(self) -> bool:
        pending = self._pending_clear_request
        self._pending_clear_request = False
        return pending

    def get_control_values(self) -> dict[str, float | bool]:
        self._deactivate_all_controls(True)
        values: dict[str, float | bool] = {
            control.spec.key: control.value for control in self.controls
        }
        values.update(
            {control.spec.key: control.value for control in self.toggle_controls}
        )
        return values

    def tick(self) -> float:
        return self.clock.tick(self.fps) / 1000.0

    def _deactivate_all_controls(
        self, commit: bool, exclude: NumericControl | None = None
    ) -> None:
        for control in self.controls:
            if control is exclude:
                continue
            control.end_drag()
            control.deactivate_text(commit)

    def _handle_control_mouse_down(self, pos: tuple[int, int]) -> bool:
        for control in self.controls:
            if control.contains_input(pos):
                self._deactivate_all_controls(True, exclude=control)
                control.activate_text()
                return True
            if control.contains_slider(pos):
                self._deactivate_all_controls(True, exclude=control)
                control.begin_drag(pos)
                return True
        for control in self.toggle_controls:
            if control.contains(pos):
                self._deactivate_all_controls(True)
                control.toggle()
                return True
        self._deactivate_all_controls(True)
        return False

    def _active_text_control(self) -> NumericControl | None:
        for control in self.controls:
            if control.active_text:
                return control
        return None

    def handle_events(self) -> bool:
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN:
                active = self._active_text_control()
                if active is not None:
                    active.handle_keydown(event)
            elif event.type == pygame.TEXTINPUT:
                active = self._active_text_control()
                if active is not None:
                    active.handle_text_input(event.text)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.play_button_rect.collidepoint(pos):
                    self._deactivate_all_controls(True)
                    if not self.simulation_loaded:
                        self._pending_start_request = True
                    else:
                        self.paused = not self.paused
                elif self.clear_button_rect.collidepoint(pos):
                    self._deactivate_all_controls(True)
                    self._pending_clear_request = True
                elif (
                    self.show_graphs
                    and self.sheep_graph is not None
                    and self.sheep_graph.is_save_button_clicked(pos)
                ):
                    self.on_save_sheep()
                    self.sheep_graph.mark_saved()
                elif (
                    self.show_graphs
                    and self.wolf_graph is not None
                    and self.wolf_graph.is_save_button_clicked(pos)
                ):
                    self.on_save_wolf()
                    self.wolf_graph.mark_saved()
                elif (
                    self.show_graphs
                    and self.grass_graph is not None
                    and self.grass_graph.is_save_button_clicked(pos)
                ):
                    self.on_save_grass()
                    self.grass_graph.mark_saved()
                else:
                    self._handle_control_mouse_down(pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for control in self.controls:
                    control.end_drag()
            elif event.type == pygame.MOUSEMOTION:
                pos = event.pos
                for control in self.controls:
                    control.update_drag(pos)
        return running

    def add_population_sample(
        self, time_sec: float, sheep_count: int, wolf_count: int, grass_count: int
    ) -> None:
        if self.sheep_graph is not None:
            self.sheep_graph.add_sample(time_sec, sheep_count)
        if self.wolf_graph is not None:
            self.wolf_graph.add_sample(time_sec, wolf_count)
        if self.grass_graph is not None:
            self.grass_graph.add_sample(time_sec, grass_count)

    def update(self, frame_dt: float) -> None:
        if self.sheep_graph is not None:
            self.sheep_graph.update(frame_dt)
        if self.wolf_graph is not None:
            self.wolf_graph.update(frame_dt)
        if self.grass_graph is not None:
            self.grass_graph.update(frame_dt)

    def _draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        bg_color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            (220, 230, 225),
            rect,
            width=1,
            border_radius=8,
        )
        text = self.small_font.render(label, True, (245, 245, 245))
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def _draw_control_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG_COLOR, self.control_panel_rect)
        pygame.draw.line(
            self.screen,
            PANEL_BORDER_COLOR,
            (self.control_panel_rect.right, 0),
            (self.control_panel_rect.right, self.height),
            2,
        )

        title = self.font.render("Simulation Controls", True, (228, 238, 233))
        self.screen.blit(title, (14, 12))

        if not self.simulation_loaded:
            status_label = "Status: Ready"
        elif self.paused:
            status_label = "Status: Paused"
        else:
            status_label = "Status: Running"
        status = self.small_font.render(status_label, True, (170, 188, 181))
        self.screen.blit(status, (16, 84))

        play_label = (
            "Play"
            if not self.simulation_loaded
            else ("Continue" if self.paused else "Pause")
        )
        play_color = (
            BUTTON_PLAY_COLOR
            if not self.simulation_loaded or self.paused
            else BUTTON_PAUSE_COLOR
        )
        self._draw_button(self.play_button_rect, play_label, play_color)
        self._draw_button(self.clear_button_rect, "Clear", BUTTON_CLEAR_COLOR)

        # hint = self.tiny_font.render(
        #    "Edit values, press Play. Clear drops the current run.",
        #    True,
        #    (146, 162, 164),
        # )
        # self.screen.blit(hint, (14, 102))

        for control in self.controls:
            control.draw(self.screen, self.tiny_font, self.small_font)
        for control in self.toggle_controls:
            control.draw(self.screen, self.small_font)

    def draw(self, world, sim_time: float, step_dt: float) -> None:
        self.screen.fill((0, 0, 0))
        self._draw_control_panel()
        pygame.draw.rect(self.screen, WORLD_BG_COLOR, self.world_rect)

        if self.show_graphs:
            pygame.draw.rect(self.screen, PANEL_BG_COLOR, self.graph_panel_rect)
            pygame.draw.line(
                self.screen,
                PANEL_BORDER_COLOR,
                (self.graph_panel_rect.x, 0),
                (self.graph_panel_rect.x, self.height),
                2,
            )
            if self.sheep_graph is not None:
                self.sheep_graph.draw(self.screen, self.font, self.small_font, sim_time)
            if self.wolf_graph is not None:
                self.wolf_graph.draw(self.screen, self.font, self.small_font, sim_time)
            if self.grass_graph is not None:
                self.grass_graph.draw(self.screen, self.font, self.small_font, sim_time)

        self.painter.draw(self.screen, world, step_dt, x_offset=self.world_x_offset)

        if world is None:
            prompt = self.font.render(
                "Press Play to start the simulation", True, (245, 245, 245)
            )
            prompt_rect = prompt.get_rect(center=self.world_rect.center)
            self.screen.blit(prompt, prompt_rect)

        fps_text = self.small_font.render(
            f"FPS: {self.clock.get_fps():5.1f}", True, (220, 230, 225)
        )
        self.screen.blit(fps_text, (self.world_rect.x + 12, self.height - 24))

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
