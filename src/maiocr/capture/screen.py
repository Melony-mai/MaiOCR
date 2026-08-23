"""Screen capture backends.

Windows: mss (fast, multi-monitor).
Linux dev fallback: grim (Wayland) / mss.
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication

from maiocr.utils.logger import get_logger

logger = get_logger()

# Let compositors redraw after the selection overlay disappears.
REGION_CAPTURE_DELAY_S = 0.15


def _grab_mss(monitor: dict) -> Image.Image:
    import mss

    with mss.mss() as sct:
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def _virtual_monitor() -> dict:
    import mss

    with mss.mss() as sct:
        return dict(sct.monitors[0])


def _monitors() -> list[dict]:
    import mss

    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors[1:]]


def _has_grim() -> bool:
    return shutil.which("grim") is not None


def _grim_capture() -> Image.Image | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output = Path(f.name)
        subprocess.run(["grim", str(output)], check=True, timeout=10)
        image = Image.open(output)
        image.load()
        output.unlink(missing_ok=True)
        return image
    except Exception as e:
        logger.warning("grim capture failed: {}", e)
        return None


def quick_capture() -> Image.Image:
    """Capture the entire virtual screen (all monitors)."""
    start = time.time()

    image = _grim_capture() if _has_grim() else _grab_mss(_virtual_monitor())

    elapsed = (time.time() - start) * 1000
    logger.info("Quick capture {} px in {:.1f} ms", image.size, elapsed)
    return image


def region_capture(rect: QRect) -> Image.Image | None:
    """
    Capture a Qt logical global-coordinate rect in physical pixels.

    Returns None when the rect is empty or no matching monitor is found.
    """
    if rect is None or rect.isEmpty():
        logger.warning("Region capture skipped: empty rect")
        return None

    time.sleep(REGION_CAPTURE_DELAY_S)
    start = time.time()
    image = _region_capture_mss(rect)

    if image is None and _has_grim():
        image = _region_capture_linux(rect)

    elapsed = (time.time() - start) * 1000
    if image is not None:
        logger.info(
            "Region capture {} -> {} px in {:.1f} ms", rect.getRect(), image.size, elapsed
        )
    else:
        logger.error("Region capture failed for rect {}", rect.getRect())
    return image


def _region_capture_mss(rect: QRect) -> Image.Image | None:
    monitor, origin, dpr = _match_monitor(rect.center())
    if monitor is None:
        return None

    left = round((rect.left() - origin.x()) * dpr)
    top = round((rect.top() - origin.y()) * dpr)
    width = max(1, round(rect.width() * dpr))
    height = max(1, round(rect.height() * dpr))

    clipped = {
        "left": monitor["left"] + left,
        "top": monitor["top"] + top,
        "width": min(width, monitor["width"] - left),
        "height": min(height, monitor["height"] - top),
    }
    if clipped["width"] <= 0 or clipped["height"] <= 0:
        logger.error("Region {} outside monitor bounds {}", clipped, monitor)
        return None

    return _grab_mss(clipped)


def _match_monitor(point) -> tuple[dict | None, object, float]:
    """Find the mss monitor whose Qt screen contains the logical point."""
    app = QGuiApplication.instance()
    if app is None:
        return None, None, 1.0

    screen = app.screenAt(point)
    if screen is None:
        # Point may sit on an edge; fall back to nearest screen.
        best, best_d = None, None
        for s in app.screens():
            d = (
                s.geometry().center() - point
            ).manhattanLength()
            if best_d is None or d < best_d:
                best, best_d = s, d
        screen = best

    if screen is None:
        return None, None, 1.0

    dpr = screen.devicePixelRatio()
    origin = screen.geometry().topLeft()
    phys_origin = origin * dpr

    target = (round(phys_origin.x()), round(phys_origin.y()))
    for monitor in _monitors():
        if (monitor["left"], monitor["top"]) == target:
            return monitor, origin, dpr

    logger.warning(
        "No mss monitor at physical origin {}, falling back to first", target
    )
    monitors = _monitors()
    return (monitors[0] if monitors else None), origin, dpr


def _region_capture_linux(rect: QRect) -> Image.Image | None:
    """Best-effort Linux dev path: full grab then crop by primary DPR."""
    full = _grim_capture()
    if full is None:
        return None

    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen else 1.0
    origin = screen.geometry().topLeft() if screen else QRect().topLeft()

    left = round((rect.left() - origin.x()) * dpr)
    top = round((rect.top() - origin.y()) * dpr)
    width = max(1, round(rect.width() * dpr))
    height = max(1, round(rect.height() * dpr))

    box = (
        max(0, min(left, full.width - 1)),
        max(0, min(top, full.height - 1)),
        min(left + width, full.width),
        min(top + height, full.height),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return full.crop(box)
