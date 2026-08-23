"""Manual smoke test: screen capture (run on a machine with a display)."""

from maiocr.capture.screen import quick_capture


def main():
    image = quick_capture()
    output = "screenshot.png"
    image.save(output)
    print(f"saved: {output} {image.size}")


if __name__ == "__main__":
    main()
