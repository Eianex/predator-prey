import csv
from pathlib import Path


class PopulationRecorder:
    def __init__(
        self, initial_time: float, sheep_count: int, wolf_count: int, grass_count: int
    ):
        self.csv_path = Path("data.csv")
        self.samples: list[tuple[float, int, int, int]] = [
            (initial_time, int(grass_count), int(sheep_count), int(wolf_count))
        ]

    def add_sample(
        self, time_sec: float, sheep_count: int, wolf_count: int, grass_count: int
    ) -> None:
        self.samples.append(
            (time_sec, int(grass_count), int(sheep_count), int(wolf_count))
        )

    def save_all(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Time", "Plants", "Sheep", "Wolves"])
            for time_sec, grass_count, sheep_count, wolf_count in self.samples:
                writer.writerow(
                    [
                        f"{time_sec:.6f}",
                        grass_count,
                        sheep_count,
                        wolf_count,
                    ]
                )

    def save_sheep(self) -> None:
        self.save_all()

    def save_wolf(self) -> None:
        self.save_all()

    def save_grass(self) -> None:
        self.save_all()
