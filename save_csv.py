import csv
from pathlib import Path


class PopulationRecorder:
    def __init__(
        self, initial_time: float, sheep_count: int, wolf_count: int, grass_count: int
    ):
        self.sheep_csv_path = Path("data_sheep.csv")
        self.wolf_csv_path = Path("data_wolf.csv")
        self.grass_csv_path = Path("data_grass.csv")

        self.sheep_samples: list[tuple[float, float]] = [
            (initial_time, float(sheep_count))
        ]
        self.wolf_samples: list[tuple[float, float]] = [
            (initial_time, float(wolf_count))
        ]
        self.grass_samples: list[tuple[float, float]] = [
            (initial_time, float(grass_count))
        ]

    def add_sample(
        self, time_sec: float, sheep_count: int, wolf_count: int, grass_count: int
    ) -> None:
        self.sheep_samples.append((time_sec, float(sheep_count)))
        self.wolf_samples.append((time_sec, float(wolf_count)))
        self.grass_samples.append((time_sec, float(grass_count)))

    @staticmethod
    def _write_csv(path: Path, samples: list[tuple[float, float]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time_sec", "population"])
            for t, v in samples:
                writer.writerow([f"{t:.6f}", int(round(v))])

    def save_sheep(self) -> None:
        PopulationRecorder._write_csv(self.sheep_csv_path, self.sheep_samples)

    def save_wolf(self) -> None:
        PopulationRecorder._write_csv(self.wolf_csv_path, self.wolf_samples)

    def save_grass(self) -> None:
        PopulationRecorder._write_csv(self.grass_csv_path, self.grass_samples)

    def save_all(self) -> None:
        self.save_sheep()
        self.save_wolf()
        self.save_grass()
