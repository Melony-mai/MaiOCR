"""Generate resources/maiocr.ico (multi-size Windows icon).

Run from the project root:
    python scripts/generate_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZES = [256, 128, 64, 48, 32, 16]
BG = "#2d7dd2"
EDGE = "#1b5e9e"


def render(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = max(2, size * 12 // 64)
    inset = max(1, size // 32)
    draw.rounded_rectangle(
        (inset, inset, size - inset - 1, size - inset - 1),
        radius=radius,
        fill=BG,
        outline=EDGE,
        width=max(1, size // 32),
    )

    # Letter M centred; fall back to a plain bar motif at tiny sizes.
    if size >= 32:
        font = None
        for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, size * 46 // 100)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox(
            (0, 0), "M", font=font
        )
        w, h = right - left, bottom - top
        draw.text(
            ((size - w) // 2 - left, (size - h) // 2 - top),
            "M",
            font=font,
            fill="white",
        )
    else:
        bar_w = max(1, size // 6)
        x0 = (size - bar_w) // 2
        draw.rectangle((x0, size // 4, x0 + bar_w, 3 * size // 4), fill="white")
    return img


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "resources"
    out_dir.mkdir(exist_ok=True)

    frames = [render(s) for s in SIZES]
    out = out_dir / "maiocr.ico"
    frames[0].save(out, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
