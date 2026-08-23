import sys

from maiocr.hotkey.base import HotkeyManager
from maiocr.utils.logger import get_logger

logger = get_logger()


def create_hotkey_manager(**kwargs) -> HotkeyManager:
    """
    Create a hotkey manager for the current platform.

    Windows: Win+PrtSc / Win+Ctrl+PrtSc via RegisterHotKey.
    Linux (dev only): Shift+Enter via evdev, text mode only.
    """
    if sys.platform == "win32":
        from maiocr.hotkey.windows import WindowsHotkeyManager

        return WindowsHotkeyManager(**kwargs)

    if sys.platform == "linux":
        from maiocr.hotkey.linux import LinuxHotkeyManager

        return LinuxHotkeyManager(**kwargs)

    raise NotImplementedError(f"Unsupported platform: {sys.platform}")
