"""Manual smoke test: full pipeline (capture -> OCR -> post-process)."""

import sys

from maiocr.core.pipeline import MaiOCRPipeline, Mode
from maiocr.utils.clipboard import copy_text
from maiocr.utils.logger import get_logger

logger = get_logger()


def main():
    mode = Mode.CODE if "--code" in sys.argv else Mode.TEXT
    region_mode = "--region" in sys.argv

    pipe = MaiOCRPipeline()
    result = pipe.run(mode=mode)

    if result.output.strip():
        result.copied = copy_text(result.output)

    print(f"summary   : {result.summary}")
    print(f"chars     : {len(result.output)}")
    print(f"copied    : {result.copied}")
    print(
        f"timing ms : capture={result.capture_ms} ocr={result.ocr_ms} "
        f"total={result.total_ms}"
    )
    if region_mode:
        print("(note: --region flag is handled by the GUI, not this script)")
    print("--- output ---")
    print(result.output)


if __name__ == "__main__":
    main()
