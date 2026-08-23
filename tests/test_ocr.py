"""Manual smoke test: OCR engine on a local image."""

import sys
from pathlib import Path

from PIL import Image

from maiocr.ocr.engine import OCREngine


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshot.png")
    if not path.exists():
        print(f"not found: {path}")
        return

    engine = OCREngine()
    result = engine.recognize(Image.open(path))

    print(f"lines: {len(result.lines)}, mean score: {result.mean_score:.3f}")
    print("====== OCR RESULT ======")
    print(result.text)


if __name__ == "__main__":
    main()
