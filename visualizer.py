from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator


BACKGROUND_COLOR = "#f4f1ea"
PANEL_COLOR = "#ece7dc"
GRID_COLOR = "#d5cec2"
TEXT_COLOR = "#2b2b2b"
PLANTS_COLOR = "#6ca95f"
SHEEP_COLOR = "#a6d97d"
WOLVES_COLOR = "#d57a5e"
DEFAULT_WINDOW_SIZE = "1280x860"
DEFAULT_CSV_NAME = "data.csv"
DEFAULT_IMAGE_NAME = "graph.png"


@dataclass(frozen=True)
class Sample:
    time_sec: float
    plants: int
    sheep: int
    wolves: int


@dataclass(frozen=True)
class PlotData:
    samples: list[Sample]
    crop_start_time: float
    shifted_to_zero: bool


def read_samples(path: Path) -> list[Sample]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        samples: list[Sample] = []
        for row in reader:
            samples.append(
                Sample(
                    time_sec=float(row["Time"]),
                    plants=int(float(row["Plants"])),
                    sheep=int(float(row["Sheep"])),
                    wolves=int(float(row["Wolves"])),
                )
            )
    return samples


def write_samples(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time", "Plants", "Sheep", "Wolves"])
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.time_sec:.6f}",
                    sample.plants,
                    sample.sheep,
                    sample.wolves,
                ]
            )


def smooth_series(
    times: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(times) < 3:
        return times, values

    dense_count = max(300, len(times) * 12)
    dense_times = np.linspace(times[0], times[-1], dense_count)
    dense_values = np.interp(dense_times, times, values)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    padded_values = np.pad(dense_values, (2, 2), mode="edge")
    smoothed_values = np.convolve(padded_values, kernel, mode="same")[2:-2]
    return dense_times, smoothed_values


class VisualizerApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_csv_path: Path | None = None,
        initial_title: str = "",
        initial_crop_start: float = 0.0,
        shift_to_zero: bool = True,
    ) -> None:
        self.root = root
        self.root.title("Population Visualizer")
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.configure(bg=BACKGROUND_COLOR)

        self.loaded_path: Path | None = None
        self.samples: list[Sample] = []

        self.crop_start_var = tk.StringVar(value=f"{initial_crop_start:.0f}")
        self.shift_to_zero_var = tk.BooleanVar(value=shift_to_zero)
        self.title_var = tk.StringVar(value=initial_title)
        self.show_plants_var = tk.BooleanVar(value=True)
        self.show_sheep_var = tk.BooleanVar(value=True)
        self.show_wolves_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Open a CSV file to visualize it.")

        self._build_layout()
        self._build_plot()
        self._bind_events()

        if initial_csv_path is not None and initial_csv_path.exists():
            self.load_csv(initial_csv_path)
        else:
            default_csv_path = Path(DEFAULT_CSV_NAME)
            if default_csv_path.exists():
                self.load_csv(default_csv_path)
            else:
                self.render_plot()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.root, padding=(14, 8, 14, 6))
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(9, weight=1)

        ttk.Button(controls, text="Open CSV", command=self.open_csv_dialog).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(controls, text="Save CSV", command=self.save_csv_dialog).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Button(controls, text="Save Image", command=self.save_image_dialog).grid(
            row=0, column=2, padx=(0, 14)
        )

        ttk.Label(controls, text="Start T").grid(row=0, column=3, padx=(0, 6))
        self.crop_entry = ttk.Entry(
            controls, width=10, textvariable=self.crop_start_var
        )
        self.crop_entry.grid(row=0, column=4, padx=(0, 8))

        ttk.Button(controls, text="Apply Crop", command=self.apply_crop).grid(
            row=0, column=5, padx=(0, 8)
        )
        ttk.Button(controls, text="Reset Crop", command=self.reset_crop).grid(
            row=0, column=6, padx=(0, 14)
        )

        self.shift_check = ttk.Checkbutton(
            controls,
            text="Shift to zero",
            variable=self.shift_to_zero_var,
            command=self.render_plot,
        )
        self.shift_check.grid(row=0, column=7, padx=(0, 14))

        ttk.Label(controls, text="Title").grid(row=0, column=8, padx=(0, 6))
        self.title_entry = ttk.Entry(controls, textvariable=self.title_var)
        self.title_entry.grid(row=0, column=9, sticky="ew")

        species_controls = ttk.Frame(self.root, padding=(14, 0, 14, 1))
        species_controls.grid(row=1, column=0, sticky="ew")

        ttk.Label(species_controls, text="Show").grid(row=0, column=0, padx=(0, 10))
        ttk.Checkbutton(
            species_controls,
            text="Plants",
            variable=self.show_plants_var,
            command=self.render_plot,
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Checkbutton(
            species_controls,
            text="Sheep",
            variable=self.show_sheep_var,
            command=self.render_plot,
        ).grid(row=0, column=2, padx=(0, 10))
        ttk.Checkbutton(
            species_controls,
            text="Wolves",
            variable=self.show_wolves_var,
            command=self.render_plot,
        ).grid(row=0, column=3, padx=(0, 10))

        status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(16, 0, 16, 10),
            foreground=TEXT_COLOR,
        )
        status_label.grid(row=3, column=0, sticky="ew")

    def _build_plot(self) -> None:
        figure = Figure(figsize=(10, 7), dpi=100, facecolor=BACKGROUND_COLOR)
        grid_spec = figure.add_gridspec(2, 1, height_ratios=[7.35, 0.4], hspace=0.2)
        figure.subplots_adjust(top=0.985, bottom=0.08, left=0.08, right=0.985)
        self.figure = figure
        self.graph_axis = figure.add_subplot(grid_spec[0])
        self.histogram_axis = figure.add_subplot(grid_spec[1])

        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=2, column=0, sticky="nsew", padx=10, pady=(0, 0)
        )

    def _bind_events(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_csv_dialog())
        self.root.bind("<Control-s>", lambda _event: self.save_csv_dialog())
        self.root.bind("<Return>", lambda _event: self.apply_crop())
        self.title_var.trace_add("write", lambda *_args: self.render_plot())

    def open_csv_dialog(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Open CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not selected_path:
            return
        self.load_csv(Path(selected_path))

    def load_csv(self, path: Path) -> None:
        try:
            self.samples = read_samples(path)
        except Exception as exc:
            messagebox.showerror("Open CSV", f"Failed to load CSV file.\n\n{exc}")
            return

        self.loaded_path = path
        self.status_var.set(f"Loaded {path}")
        self.render_plot()

    def save_csv_dialog(self) -> None:
        plot_data = self.current_plot_data()
        if len(plot_data.samples) == 0:
            messagebox.showwarning("Save CSV", "There is no visible data to save.")
            return

        selected_path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialfile=DEFAULT_CSV_NAME,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not selected_path:
            return

        try:
            write_samples(Path(selected_path), plot_data.samples)
        except Exception as exc:
            messagebox.showerror("Save CSV", f"Failed to save CSV file.\n\n{exc}")
            return

        self.status_var.set(f"Saved visible data to {selected_path}")

    def save_image_dialog(self) -> None:
        plot_data = self.current_plot_data()
        if len(plot_data.samples) == 0:
            messagebox.showwarning("Save Image", "There is no visible graph to save.")
            return

        selected_path = filedialog.asksaveasfilename(
            title="Save Image",
            initialfile=DEFAULT_IMAGE_NAME,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not selected_path:
            return

        try:
            self.figure.savefig(selected_path, dpi=160, facecolor=BACKGROUND_COLOR)
        except Exception as exc:
            messagebox.showerror("Save Image", f"Failed to save image.\n\n{exc}")
            return

        self.status_var.set(f"Saved image to {selected_path}")

    def apply_crop(self) -> None:
        try:
            float(self.crop_start_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("Crop", "Start T must be a valid number.")
            return
        self.render_plot()

    def reset_crop(self) -> None:
        self.crop_start_var.set("0")
        self.render_plot()

    def current_plot_data(self) -> PlotData:
        crop_start_time = float(self.crop_start_var.get().strip() or "0")
        visible_samples = [
            sample for sample in self.samples if sample.time_sec >= crop_start_time
        ]
        shift_to_zero = self.shift_to_zero_var.get()
        if shift_to_zero and len(visible_samples) > 0:
            shifted_samples = [
                Sample(
                    time_sec=sample.time_sec - crop_start_time,
                    plants=sample.plants,
                    sheep=sample.sheep,
                    wolves=sample.wolves,
                )
                for sample in visible_samples
            ]
        else:
            shifted_samples = list(visible_samples)
        return PlotData(
            samples=shifted_samples,
            crop_start_time=crop_start_time,
            shifted_to_zero=shift_to_zero,
        )

    def _selected_species(self) -> list[tuple[str, bool, str]]:
        return [
            ("Plants", self.show_plants_var.get(), PLANTS_COLOR),
            ("Sheep", self.show_sheep_var.get(), SHEEP_COLOR),
            ("Wolves", self.show_wolves_var.get(), WOLVES_COLOR),
        ]

    def render_plot(self) -> None:
        self.graph_axis.clear()
        self.histogram_axis.clear()

        plot_data = self.current_plot_data()
        samples = plot_data.samples
        if len(samples) == 0:
            self._render_empty_state()
            self.canvas.draw_idle()
            return

        times = np.array([sample.time_sec for sample in samples], dtype=float)
        series_by_name = {
            "Plants": np.array([sample.plants for sample in samples], dtype=float),
            "Sheep": np.array([sample.sheep for sample in samples], dtype=float),
            "Wolves": np.array([sample.wolves for sample in samples], dtype=float),
        }
        visible_series = [
            (name, series_by_name[name], color)
            for name, is_visible, color in self._selected_species()
            if is_visible
        ]
        if len(visible_series) == 0:
            self._render_empty_state("No species selected. Check at least one species.")
            self.canvas.draw_idle()
            return

        self.graph_axis.set_facecolor(PANEL_COLOR)
        self.graph_axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        self.graph_axis.set_axisbelow(True)
        self.graph_axis.spines["top"].set_visible(False)
        self.graph_axis.spines["right"].set_visible(False)
        self.graph_axis.spines["left"].set_color("#938a79")
        self.graph_axis.spines["bottom"].set_color("#938a79")

        max_population = 1.0
        for name, values, color in visible_series:
            self._plot_smoothed_line(times, values, color, name)
            max_population = max(max_population, float(values.max()))

        self.graph_axis.set_ylim(0.0, max_population * 1.08)
        self.graph_axis.set_xlim(
            float(times[0]),
            float(times[-1]) if len(times) > 1 else float(times[0]) + 1.0,
        )
        self.graph_axis.set_ylabel("Population", color=TEXT_COLOR)
        self.graph_axis.set_xlabel("Time [s]", color=TEXT_COLOR, labelpad=4)
        self.graph_axis.tick_params(colors=TEXT_COLOR)
        self.graph_axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        self.graph_axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: str(int(round(value))))
        )
        self.graph_axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))

        title = self.title_var.get().strip()
        if title:
            self.graph_axis.set_title(title, color=TEXT_COLOR, pad=12)

        self._render_histogram(visible_series)
        crop_info = f"Start T={plot_data.crop_start_time:g}s"
        if plot_data.shifted_to_zero:
            crop_info += " | shifted to zero"
        visible_labels = ", ".join(name for name, _, _ in visible_series)
        self.status_var.set(
            f"{crop_info} | Visible samples: {len(samples)} | Showing: {visible_labels}"
            if self.loaded_path is None
            else f"{self.loaded_path} | {crop_info} | Visible samples: {len(samples)} | Showing: {visible_labels}"
        )
        self.canvas.draw_idle()

    def _plot_smoothed_line(
        self,
        times: np.ndarray,
        values: np.ndarray,
        color: str,
        label: str,
    ) -> None:
        smooth_times, smooth_values = smooth_series(times, values)
        self.graph_axis.plot(
            smooth_times,
            smooth_values,
            color=color,
            linewidth=2.7,
            solid_capstyle="round",
            label=label,
        )

    def _render_histogram(
        self, visible_series: list[tuple[str, np.ndarray, str]]
    ) -> None:
        labels = [name for name, _, _ in visible_series]
        colors = [color for _, _, color in visible_series]
        self.histogram_axis.bar(labels, [0.42] * len(labels), color=colors, width=0.48)
        self.histogram_axis.set_ylim(0, 0.65)
        self.histogram_axis.set_yticks([])
        self.histogram_axis.tick_params(axis="x", colors=TEXT_COLOR, labelsize=11)
        self.histogram_axis.spines["top"].set_visible(False)
        self.histogram_axis.spines["right"].set_visible(False)
        self.histogram_axis.spines["left"].set_visible(False)
        self.histogram_axis.spines["bottom"].set_visible(False)
        self.histogram_axis.set_facecolor(BACKGROUND_COLOR)

    def _render_empty_state(
        self, message: str = "No visible data. Open a CSV file or adjust Start T."
    ) -> None:
        self.graph_axis.set_facecolor(PANEL_COLOR)
        self.graph_axis.set_xticks([])
        self.graph_axis.set_yticks([])
        self.graph_axis.spines["top"].set_visible(False)
        self.graph_axis.spines["right"].set_visible(False)
        self.graph_axis.spines["left"].set_visible(False)
        self.graph_axis.spines["bottom"].set_visible(False)
        self.graph_axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=13,
            transform=self.graph_axis.transAxes,
        )
        self.histogram_axis.set_xticks([])
        self.histogram_axis.set_yticks([])
        self.histogram_axis.spines["top"].set_visible(False)
        self.histogram_axis.spines["right"].set_visible(False)
        self.histogram_axis.spines["left"].set_visible(False)
        self.histogram_axis.spines["bottom"].set_visible(False)
        self.histogram_axis.set_facecolor(BACKGROUND_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize population CSV data.")
    parser.add_argument(
        "csv_path", nargs="?", help="Optional CSV file to open on startup."
    )
    parser.add_argument("--title", default="", help="Optional graph title.")
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Initial crop start time in seconds.",
    )
    parser.add_argument(
        "--no-shift-times",
        action="store_true",
        help="Keep original time values after cropping instead of shifting them to zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_csv_path = Path(args.csv_path) if args.csv_path else None

    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TFrame", background=BACKGROUND_COLOR)
    style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
    style.configure("TCheckbutton", background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
    style.configure("TButton", padding=(10, 6))
    style.configure("TEntry", padding=(6, 4))

    VisualizerApp(
        root,
        initial_csv_path=initial_csv_path,
        initial_title=args.title,
        initial_crop_start=args.start_time,
        shift_to_zero=not args.no_shift_times,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
