import asyncio
import csv
import io
import sys
from pathlib import Path
from typing import Callable


def _write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _desktop_pick_save_path(
    *,
    title: str,
    default_name: str,
    description: str,
    extension: str,
) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        selected_path = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_name,
            defaultextension=extension,
            filetypes=[(description, f"*{extension}"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    if not selected_path:
        return None
    return Path(selected_path)


def _desktop_pick_open_path(
    *,
    title: str,
    description: str,
    extension: str,
) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        selected_path = filedialog.askopenfilename(
            title=title,
            filetypes=[(description, f"*{extension}"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    if not selected_path:
        return None
    return Path(selected_path)


def _schedule_browser_task(coro) -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print("Browser file dialog requires a running async loop.")
        return False
    loop.create_task(coro)
    return True


async def _browser_save_text(
    text: str,
    *,
    default_name: str,
    description: str,
    extension: str,
    mime_type: str,
) -> None:
    try:
        from platform import window

        picker = getattr(window, "showSaveFilePicker", None)
        if picker is not None:
            handle = await picker(
                {
                    "suggestedName": default_name,
                    "types": [
                        {
                            "description": description,
                            "accept": {mime_type: [extension]},
                        }
                    ],
                }
            )
            writable = await handle.createWritable()
            await writable.write(text)
            await writable.close()
            print(f"Saved file: {default_name}")
            return

        blob = window.Blob.new([text], {"type": mime_type})
        object_url = window.URL.createObjectURL(blob)
        document = window.document
        link = document.createElement("a")
        link.href = object_url
        link.download = default_name
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(object_url)
        print(f"Started download for: {default_name}")
    except Exception as exc:
        print(f"Save cancelled or failed: {exc}")


async def _browser_open_text(
    on_loaded: Callable[[str], None],
    *,
    description: str,
    extension: str,
    mime_type: str,
) -> None:
    try:
        from platform import window

        picker = getattr(window, "showOpenFilePicker", None)
        if picker is not None:
            handles = await picker(
                {
                    "multiple": False,
                    "types": [
                        {
                            "description": description,
                            "accept": {mime_type: [extension]},
                        }
                    ],
                }
            )
            if handles is None or len(handles) == 0:
                return
            file_obj = await handles[0].getFile()
            text = await file_obj.text()
            on_loaded(str(text))
            return

        await _browser_open_text_via_input(
            on_loaded,
            extension=extension,
        )
    except Exception as exc:
        print(f"Open cancelled or failed: {exc}")


async def _browser_open_text_via_input(
    on_loaded: Callable[[str], None],
    *,
    extension: str,
) -> None:
    from platform import window

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str | None] = loop.create_future()
    document = window.document
    input_element = document.createElement("input")
    input_element.type = "file"
    input_element.accept = extension
    input_element.style.display = "none"
    document.body.appendChild(input_element)

    async def _consume_selection() -> None:
        try:
            files = input_element.files
            if files is None or getattr(files, "length", 0) == 0:
                if not future.done():
                    future.set_result(None)
                return
            file_obj = files.item(0)
            if file_obj is None:
                if not future.done():
                    future.set_result(None)
                return
            text = await file_obj.text()
            if not future.done():
                future.set_result(str(text))
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    def _on_change(_event) -> None:
        loop.create_task(_consume_selection())

    input_element.onchange = _on_change
    input_element.click()

    try:
        text = await future
    finally:
        document.body.removeChild(input_element)

    if text is not None:
        on_loaded(text)


def save_text_with_dialog(
    text: str,
    *,
    title: str,
    default_name: str,
    description: str,
    extension: str,
    mime_type: str,
    fallback_path: Path | None = None,
    use_dialog: bool = True,
) -> Path | None:
    if not use_dialog:
        path = fallback_path if fallback_path is not None else Path(default_name)
        _write_text_file(path, text)
        return path

    if sys.platform == "emscripten":
        _schedule_browser_task(
            _browser_save_text(
                text,
                default_name=default_name,
                description=description,
                extension=extension,
                mime_type=mime_type,
            )
        )
        return None

    selected_path = _desktop_pick_save_path(
        title=title,
        default_name=default_name,
        description=description,
        extension=extension,
    )
    if selected_path is None:
        return None

    _write_text_file(selected_path, text)
    print(f"Saved file: {selected_path}")
    return selected_path


def open_text_with_dialog(
    *,
    title: str,
    description: str,
    extension: str,
    mime_type: str,
    on_loaded: Callable[[str], None] | None = None,
) -> str | None:
    if sys.platform == "emscripten":
        if on_loaded is None:
            print("Browser import requires a callback.")
            return None
        _schedule_browser_task(
            _browser_open_text(
                on_loaded,
                description=description,
                extension=extension,
                mime_type=mime_type,
            )
        )
        return None

    selected_path = _desktop_pick_open_path(
        title=title,
        description=description,
        extension=extension,
    )
    if selected_path is None:
        return None

    text = selected_path.read_text(encoding="utf-8")
    print(f"Imported file: {selected_path}")
    return text


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

    def to_csv_text(self) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
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
        return buffer.getvalue()

    def save_all(self, *, use_dialog: bool = True, path: Path | None = None) -> None:
        csv_text = self.to_csv_text()
        save_text_with_dialog(
            csv_text,
            title="Save CSV",
            default_name=self.csv_path.name,
            description="CSV files",
            extension=".csv",
            mime_type="text/csv",
            fallback_path=self.csv_path if path is None else path,
            use_dialog=use_dialog,
        )

    def save_sheep(self) -> None:
        self.save_all()

    def save_wolf(self) -> None:
        self.save_all()

    def save_grass(self) -> None:
        self.save_all()
