"""Launch-at-login persistence (per-user registry Run key on Windows)."""

import subprocess
import sys
from pathlib import Path

from maiocr.utils.logger import get_logger
from maiocr.utils.paths import APP_NAME, app_dir

logger = get_logger()

if sys.platform == "win32":
    import winreg

_RUN_SUBKEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = APP_NAME


def is_supported() -> bool:
    return sys.platform == "win32"


def launch_command() -> list[str]:
    """Command line that starts this application again.

    Frozen EXE: the executable itself. Source tree: current interpreter
    plus the project's main.py.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, str(app_dir() / "main.py")]


def command_line() -> str:
    """Registry-ready quoted command line (Run values are plain strings)."""
    return subprocess.list2cmdline(launch_command())


def workdir() -> Path:
    """Directory the relaunched process should start in."""
    return Path(launch_command()[-1]).resolve().parent


def spawn_relaunch() -> None:
    """Start a detached new application process (restart support)."""
    cmd = launch_command()
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    subprocess.Popen(cmd, cwd=str(workdir()), **kwargs)
    logger.info("MaiOCR relaunched: {}", cmd)


def is_enabled() -> bool:
    if not is_supported():
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_SUBKEY) as key:
            value, _type = winreg.QueryValueEx(key, _VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.error("Autostart query failed: {}", e)
        return False


def set_enabled(enabled: bool) -> str:
    """Enable or disable launch at login; returns the registered line.

    Raises OSError when the registry cannot be updated so callers can
    revert their UI state.
    """
    if not is_supported():
        raise OSError("autostart is only supported on Windows")
    line = command_line()
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_SUBKEY) as key:
            if enabled:
                winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, line)
            else:
                try:
                    winreg.DeleteValue(key, _VALUE_NAME)
                except FileNotFoundError:
                    pass
    except OSError as e:
        logger.error("Autostart update failed (enabled={}): {}", enabled, e)
        raise
    logger.info("Autostart {} ({})", "enabled" if enabled else "disabled", line)
    return line
