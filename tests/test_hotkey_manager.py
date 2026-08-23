"""Manual smoke test: hotkey listener (Linux dev machines only)."""

from maiocr.hotkey.manager import create_hotkey_manager
from maiocr.utils.logger import get_logger

logger = get_logger()


def on_text():
    print("=== TEXT HOTKEY TRIGGERED ===")


def on_code():
    print("=== CODE HOTKEY TRIGGERED ===")


if __name__ == "__main__":
    hotkey = create_hotkey_manager()
    hotkey.start({"text": on_text, "code": on_code})
    print("Listening... Ctrl+C to quit")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hotkey.stop()
