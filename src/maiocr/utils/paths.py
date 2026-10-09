"""Application data paths, frozen-EXE aware."""

import os
import sys
from pathlib import Path

APP_NAME = "MaiOCR"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """Directory of the running executable (or source tree)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def app_data_dir() -> Path:
    """
    Writable per-user data directory.

    Windows: %LOCALAPPDATA%/MaiOCR
    Others:  $XDG_DATA_HOME/MaiOCR (default ~/.local/share/MaiOCR)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        path = base / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        path = base / APP_NAME

    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_file_path() -> Path:
    """Path to the persistent history JSON file."""
    return app_data_dir() / "history.json"



def external_models_dir() -> Path:
    """
    Optional user-supplied model directory.

    Checked locations (first existing hit wins):
      - <app_dir>/models          (next to the EXE)
      - <app_data_dir>/models
    """
    candidates = [
        app_dir() / "models",
        app_data_dir() / "models",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def icon_path() -> Path | None:
    """Application icon file, or None when not bundled."""
    candidates = [
        app_dir() / "resources" / "MaiOCR.ico",
        app_dir() / "MaiOCR.ico",          # project-root copy (dev tree)
        app_dir() / "resources" / "icon.ico",
        app_dir() / "icon.ico",
    ]
    if is_frozen():
        # PyInstaller >= 6 onedir keeps data files inside <exe_dir>/_internal
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.insert(0, Path(meipass) / "resources" / "icon.ico")
            candidates.append(Path(meipass) / "resources" / "maiocr.ico")
    for path in candidates:
        if path.is_file():
            return path
    return None
