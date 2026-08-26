import time
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image

from maiocr.capture.screen import quick_capture, region_capture
from maiocr.ocr.code import CodeOcrResult, process_code, render_layout
from maiocr.ocr.engine import OCREngine
from maiocr.utils.logger import get_logger

logger = get_logger()


class Mode(str, Enum):
    TEXT = "text"
    CODE = "code"


@dataclass
class PipelineResult:
    mode: Mode
    raw_text: str
    output: str                 # clipboard content (plain text or markdown)
    language: str = ""
    is_code: bool = False
    copied: bool = False        # set by the UI thread after copying
    capture_ms: float = 0.0
    ocr_ms: float = 0.0
    total_ms: float = 0.0
    structure: dict | None = None   # code-mode structural summary

    @property
    def summary(self) -> str:
        if self.mode is Mode.CODE and self.is_code:
            base = f"代码识别 · {self.language or '未知语言'}"
            s = self.structure or {}
            bits = []
            if s.get("functions"):
                bits.append(f"{s['functions']}函数")
            if s.get("classes"):
                bits.append(f"{s['classes']}类")
            if s.get("imports"):
                bits.append(f"{s['imports']}导入")
            if s.get("balanced") is False:
                bits.append("括号不闭合")
            if bits:
                base += " · " + "/".join(bits)
            return base
        return "文字识别"


@dataclass
class MaiOCRPipeline:
    """
    Capture -> OCR -> post-process.

    Pure processing only: clipboard copying is intentionally NOT done here
    because run() executes on worker threads while Qt clipboards must be
    touched from the GUI thread (the app layer owns that step).
    """

    _engine: OCREngine = field(default_factory=OCREngine, repr=False)

    def __post_init__(self):
        # Engine initialization is now deferred to first use to keep VRAM free.
        # Warmup will run lazily on the first inference.
        pass

    @property
    def acceleration(self) -> str:
        """Human-readable acceleration mode, e.g. 'GPU/DirectML'."""
        return self._engine.mode_label

    @property
    def vram_status(self) -> str:
        """Current VRAM status text for display."""
        return self._engine.vram_status

    @property
    def vram_in_use(self) -> bool:
        """Whether VRAM is currently allocated for GPU inference."""
        return self._engine.vram_in_use

    def release_vram(self) -> bool:
        """Manually release GPU/VRAM resources. Returns True if VRAM was released."""
        return self._engine.release_vram()

    def run(
        self,
        mode: Mode = Mode.TEXT,
        image: Image.Image | None = None,
        region=None,
    ) -> PipelineResult:
        """Capture (unless an image/region is given) -> OCR -> post-process."""
        logger.info("Pipeline started (mode={})", mode.value)
        t0 = time.time()

        capture_ms = 0.0
        if image is None:
            if region is not None:
                ts = time.time()
                image = region_capture(region)
                capture_ms = (time.time() - ts) * 1000
                if image is None:
                    raise RuntimeError("区域截图失败，请重试")
            else:
                ts = time.time()
                image = quick_capture()
                capture_ms = (time.time() - ts) * 1000

        ts = time.time()
        ocr = self._engine.recognize(image)
        ocr_ms = (time.time() - ts) * 1000

        raw_text = ocr.text
        layout_text = ""

        result = PipelineResult(
            mode=mode,
            raw_text=raw_text,
            output="",
            capture_ms=round(capture_ms, 1),
            ocr_ms=round(ocr_ms, 1),
        )

        if not raw_text.strip():
            result.total_ms = round((time.time() - t0) * 1000, 1)
            logger.warning("Pipeline produced no text")
            return result

        if mode is Mode.CODE:
            # Geometry-aware reconstruction: OCR text carries no leading
            # whitespace; indentation and blank lines are rebuilt from the
            # detected bounding boxes. Falls back to flat text without them.
            if ocr.geom:
                layout_text = render_layout(ocr.geom)
                raw_text = layout_text or raw_text
                levels = {
                    len(line) - len(line.lstrip(" "))
                    for line in layout_text.splitlines()
                    if line.strip()
                }
                logger.info(
                    "Code layout rebuilt from geometry: {} rows, indent levels={}",
                    len(ocr.geom),
                    sorted(levels)[:8],
                )
            processed: CodeOcrResult = process_code(raw_text)
            result.output = processed.output
            result.language = processed.language
            result.is_code = processed.is_code
            result.structure = processed.structure or None
            if processed.artifacts.get("gutter_stripped"):
                logger.info("Code mode: line-number gutter stripped")
            if processed.artifacts.get("prompts_stripped"):
                logger.info(
                    "Code mode: {} prompt(s) stripped",
                    processed.artifacts["prompts_stripped"],
                )
        else:
            result.output = raw_text

        result.total_ms = round((time.time() - t0) * 1000, 1)

        logger.info(
            "Pipeline finished: mode={} chars={} code={} lang={} "
            "capture={}ms ocr={}ms total={}ms",
            mode.value,
            len(result.output),
            result.is_code,
            result.language or "-",
            result.capture_ms,
            result.ocr_ms,
            result.total_ms,
        )
        return result
