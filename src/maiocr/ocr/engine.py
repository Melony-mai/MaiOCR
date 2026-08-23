import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maiocr.utils.logger import get_logger
from maiocr.utils.paths import external_models_dir

logger = get_logger()

# Kana share of CJK chars above which we re-run with the Japanese model.
KANA_RATIO_THRESHOLD = 0.12

_BARE_NUMBER_RX = re.compile(r"^\d{1,4}[.:|\)]?$")


@dataclass
class GeomLine:
    """
    One visual OCR line with its page geometry (pixels, origin top-left).

    ``left`` is what indentation is reconstructed from; ``top``/``bottom``
    drive blank-line detection and same-line fragment merging.
    """

    text: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


def drop_gutter_boxes(entries: list[GeomLine]) -> tuple[list[GeomLine], int]:
    """
    Remove editor line-number boxes *before* rows are merged.

    A screenshot's gutter shows up as standalone numeric boxes in a narrow
    column left of the code. If they are merged into rows first, every
    row's ``left`` becomes the gutter position and the true indentation is
    lost. Qualification for the gutter column:

      - box text is a bare number (optionally with ``. : | )`` suffix)
      - the box lies entirely left of every non-numeric box
      - values increase strictly from top to bottom (gaps allowed - OCR
        regularly misses some numbers)

    Returns (remaining entries, number of dropped gutter boxes).
    """
    numeric = [e for e in entries if _BARE_NUMBER_RX.match(e.text.strip())]
    if len(numeric) < 2:
        return entries, 0

    main_left = min(
        (e.left for e in entries if e not in numeric), default=None
    )
    if main_left is None:
        return entries, 0

    column = [e for e in numeric if e.right <= main_left]
    if len(column) != len(numeric):
        return entries, 0

    column.sort(key=lambda e: e.top)
    nums = [int(re.sub(r"\D", "", e.text)) for e in column]
    if not all(b > a for a, b in zip(nums, nums[1:])):
        return entries, 0

    dropped_ids = {id(e) for e in column}
    remaining = [e for e in entries if id(e) not in dropped_ids]
    logger.info("Dropped {} line-number gutter box(es)", len(column))
    return remaining, len(column)


def merge_fragment_lines(entries: list[GeomLine]) -> list[GeomLine]:
    """
    Merge boxes belonging to the same visual row and order rows top-down.

    Detectors occasionally split one logical line into several horizontally
    adjacent boxes; those must be concatenated before any layout work.
    """
    if not entries:
        return []
    med_h = sorted(e.bottom - e.top for e in entries)[len(entries) // 2]

    ordered = sorted(entries, key=lambda e: (e.cy, e.left))
    rows: list[list[GeomLine]] = []
    for e in ordered:
        if rows:
            row_cy = sum(x.cy for x in rows[-1]) / len(rows[-1])
            if abs(e.cy - row_cy) <= 0.6 * max(med_h, 1):
                rows[-1].append(e)
                continue
        rows.append([e])

    merged: list[GeomLine] = []
    for row in rows:
        row.sort(key=lambda e: e.left)
        parts: list[str] = []
        prev: GeomLine | None = None
        for frag in row:
            if prev is not None and frag.left - prev.right > 0.3 * max(med_h, 1):
                parts.append(" ")
            parts.append(frag.text.strip())
            prev = frag
        merged.append(
            GeomLine(
                text="".join(parts),
                left=min(f.left for f in row),
                top=min(f.top for f in row),
                right=max(f.right for f in row),
                bottom=max(f.bottom for f in row),
            )
        )
    return merged


@dataclass
class OcrResult:
    lines: list[str]
    scores: list[float]
    geom: list[GeomLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


# RapidOCR param keys per accelerator, in order of preference.
_ACCEL_PROVIDERS = (
    ("cuda", {"EngineConfig.onnxruntime.use_cuda": True}, "GPU/CUDA"),
    ("dml", {"EngineConfig.onnxruntime.use_dml": True}, "GPU/DirectML"),
)

# Consecutive GPU inference failures tolerated before degrading to CPU.
_MAX_GPU_RECOVERIES = 2

# Lone punctuation rows (``"""``, ``{``) often score below the default
# confidence gate and vanish; 0.25 keeps them visible to the repair passes.
_OCR_TEXT_SCORE = 0.25

_PROVIDER_KEYS = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
}

MODE_CPU = "CPU"

# Warmup above these thresholds means DirectML likely landed on a
# software/virtual adapter or a crippled device: tiny images compile fine
# there, but real-sized workloads fall back per-op and crawl.
_WARMUP_SLOW_MS = 1500
_WARMUP_STRESS_SLOW_MS = 4500
_DML_PROBE_DEVICE_IDS = (0, 1, 2, 3)


_RAPIDOCR_EP_PATCHED = False


def _ensure_rapidocr_plain_dict_ep_cfg():
    """
    Work around a RapidOCR bug: user-supplied EP option dicts are stored in
    an OmegaConf DictConfig, which onnxruntime rejects (isinstance(dict)),
    silently degrading every session to CPU. Normalize to a plain dict.
    """
    global _RAPIDOCR_EP_PATCHED
    if _RAPIDOCR_EP_PATCHED:
        return
    _RAPIDOCR_EP_PATCHED = True
    try:
        from rapidocr.inference_engine.onnxruntime.provider_config import (
            ProviderConfig,
        )

        original = ProviderConfig.dml_ep_cfg

        def plain_dict(self):
            value = getattr(self.cfg, "dml_ep_cfg", None)
            if value is not None:
                return dict(value)
            return original(self)

        ProviderConfig.dml_ep_cfg = plain_dict
        logger.debug("Patched RapidOCR ProviderConfig.dml_ep_cfg for OmegaConf")
    except Exception as e:
        logger.warning("Could not patch RapidOCR EP config handling: {}", e)


def detect_acceleration() -> tuple[str, dict, str]:
    """
    Pick the fastest execution provider the installed onnxruntime offers.

    Returns (mode, rapidocr params, log label). Falls back to CPU when no
    GPU provider is present (e.g. the plain CPU onnxruntime build).
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        logger.warning("onnxruntime import failed: {}", e)
        return "cpu", {}, MODE_CPU

    providers = ort.get_available_providers()
    base = {"Global.text_score": _OCR_TEXT_SCORE}
    for mode, params, label in _ACCEL_PROVIDERS:
        if _PROVIDER_KEYS[mode] in providers:
            merged = {**base, **params}
            logger.info(
                "ONNX Runtime {}: GPU acceleration via {} ({})",
                ort.__version__,
                _PROVIDER_KEYS[mode],
                providers,
            )
            return mode, merged, label

    logger.info(
        "ONNX Runtime {}: no GPU provider in {}, using CPU",
        ort.__version__,
        providers,
    )
    return "cpu", dict(base), MODE_CPU


def _session_providers(engine) -> list[str] | None:
    """Best-effort read of the active execution providers."""
    for attr_chain in (
        ("text_det", "session", "session"),
        ("text_rec", "session", "session"),
    ):
        obj = engine
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            return list(obj.get_providers())
        except AttributeError:
            continue
        except Exception:
            continue
    return None


def find_japan_model() -> Path | None:
    """
    Locate an optional offline Japanese rec model.

    Expected file (PaddleOCR PP-OCRv4 japan mobile, ONNX):
      <models_dir>/japan_PP-OCRv4_rec_mobile.onnx
    Never downloads anything - purely local lookup.
    """
    base = external_models_dir()
    names = [
        "japan_PP-OCRv4_rec_mobile.onnx",
        "japan_rec.onnx",
    ]
    for name in names:
        path = base / name
        if path.is_file():
            logger.info("Japanese rec model found: {}", path)
            return path

    # Accept any clearly-named japan rec model dropped into models/.
    for pattern in ("japan*.onnx", "*japan*rec*.onnx"):
        matches = sorted(base.glob(pattern))
        if matches:
            logger.info("Japanese rec model found: {}", matches[0])
            return matches[0]
    return None


def japan_dict_path(model_path: Path) -> Path | None:
    """Optional CTC dictionary placed next to the japan model (same stem)."""
    candidate = model_path.with_suffix(".txt")
    if candidate.is_file():
        logger.info("Japanese dict found: {}", candidate)
        return candidate
    return None


@contextmanager
def _forbid_downloads():
    """
    Hard offline guarantee around optional-engine construction.

    RapidOCR transparently downloads missing models/dicts from the network.
    We keep its skip-if-file-exists behaviour but refuse any real download,
    so MaiOCR can never go online - no matter which local files are missing.
    """
    import rapidocr.utils.download_file as df

    # Keep the original classmethod bound so skip-if-exists still works.
    original_run = df.DownloadFile.run

    def guarded(_cls, input_params):
        if not Path(input_params.save_path).exists():
            raise RuntimeError(
                f"offline mode: refusing to download {input_params.file_url}"
            )
        return original_run(input_params)

    df.DownloadFile.run = classmethod(guarded)
    try:
        yield
    finally:
        df.DownloadFile.run = classmethod(original_run.__func__)


class OCREngine:
    """
    RapidOCR wrapper.

    - Uses the fastest available GPU provider (CUDA or DirectML) and falls
      back to CPU if GPU init/inference fails.
    - Optional Japanese pass when the result looks kana-heavy and a local
      japan rec model is present.
    """

    def __init__(self):
        self._engine = None
        self._japan_engine = None
        self._japan_model_path: Path | None = find_japan_model()
        self._gpu_failures = 0

        self._mode, self._accel_params, self._label = detect_acceleration()
        self._engine = self._create_engine()

    # ------------------------------------------------------------- creation

    def _create_engine(self):
        from rapidocr import RapidOCR

        _ensure_rapidocr_plain_dict_ep_cfg()
        params = dict(self._accel_params)

        start = time.time()
        try:
            engine = RapidOCR(params=params)
        except Exception as e:
            if not params:
                raise
            logger.warning(
                "GPU engine init failed ({}), falling back to CPU", e
            )
            self._accel_params = {}
            self._mode, self._label = "cpu", MODE_CPU
            engine = RapidOCR(params={})

        elapsed = (time.time() - start) * 1000
        providers = _session_providers(engine)
        logger.info(
            "OCR engine ready [{}] in {:.0f} ms, providers={}",
            self._label,
            elapsed,
            providers or "unknown",
        )
        return engine

    def _ensure_cpu_fallback(self, error: Exception):
        """
        Recover from an inference failure.

        Transient GPU faults are common in long-running sessions (sleep/
        resume, driver hiccups, VRAM pressure). Rebuild the GPU engine up
        to ``_MAX_GPU_RECOVERIES`` consecutive times; only after that
        degrade to the CPU engine for the rest of the session.
        """
        had_gpu = bool(self._accel_params)
        if had_gpu and self._gpu_failures < _MAX_GPU_RECOVERIES:
            self._gpu_failures += 1
            logger.warning(
                "GPU inference failed ({}); rebuilding GPU engine "
                "(recovery {}/{})",
                error,
                self._gpu_failures,
                _MAX_GPU_RECOVERIES,
            )
            try:
                self._engine = self._create_engine()
                return
            except Exception as e:
                logger.warning("GPU engine rebuild failed: {}", e)
                self._accel_params = {}
                self._mode, self._label = "cpu", MODE_CPU

        if had_gpu:
            logger.error(
                "GPU unavailable after {} recovery attempt(s) - staying on "
                "CPU until restart",
                self._gpu_failures,
            )
        self._accel_params = {}
        self._mode, self._label = "cpu", MODE_CPU
        try:
            self._engine = self._create_engine()
        except Exception as e:
            logger.error("CPU fallback also failed: {}", e)
            raise

    # ------------------------------------------------------------ inference

    def recognize(self, image: Image.Image) -> OcrResult:
        result = self._run(self._engine, image)
        self._gpu_failures = 0  # healthy run resets the recovery budget

        if self._looks_japanese(result.text):
            result = self._maybe_rerun_japanese(image, result)

        logger.info(
            "OCR done: {} lines, mean score {:.3f}",
            len(result.lines),
            result.mean_score,
        )
        return result

    def _run(self, engine, image: Image.Image) -> OcrResult:
        start = time.time()
        try:
            raw = engine(np.asarray(image.convert("RGB")))
        except Exception as e:
            if engine is not self._engine:
                raise
            self._ensure_cpu_fallback(e)
            raw = self._engine(np.asarray(image.convert("RGB")))

        elapsed = (time.time() - start) * 1000

        if raw is None or raw.txts is None:
            logger.warning("OCR returned no text ({:.1f} ms)", elapsed)
            return OcrResult(lines=[], scores=[])

        lines = list(raw.txts)
        scores = (
            [float(s) for s in raw.scores]
            if raw.scores is not None
            else [1.0] * len(lines)
        )

        geom: list[GeomLine] = []
        boxes = getattr(raw, "boxes", None)
        if boxes is not None:
            try:
                n_boxes = len(boxes)
            except TypeError:
                boxes = None
                n_boxes = 0
            if n_boxes == len(lines):
                for box, txt in zip(boxes, lines):
                    try:
                        xs = [int(p[0]) for p in box]
                        ys = [int(p[1]) for p in box]
                    except (TypeError, ValueError, IndexError):
                        continue
                    if txt.strip():
                        geom.append(
                            GeomLine(
                                text=txt,
                                left=min(xs),
                                top=min(ys),
                                right=max(xs),
                                bottom=max(ys),
                            )
                        )
        # Gutter numbers must go before row merging, otherwise every row's
        # left edge becomes the gutter position and indentation is masked.
        geom, dropped = drop_gutter_boxes(geom)
        if dropped:
            logger.info("Dropped {} gutter box(es) from OCR geometry", dropped)
        geom = merge_fragment_lines(geom)

        logger.debug("Raw OCR: {} lines in {:.1f} ms", len(lines), elapsed)
        return OcrResult(lines=lines, scores=scores, geom=geom)

    @property
    def mode_label(self) -> str:
        """Human-readable acceleration mode, e.g. 'GPU/DirectML' or 'CPU'."""
        return self._label

    def warmup(self) -> float | None:
        """
        Run throwaway inferences so GPU kernels compile and buffers
        allocate before the first real capture.

        DirectML/CUDA compile kernels lazily per input shape; without this
        the first user-facing OCR pays a multi-second compilation cost.

        Doubles as a health check: a tiny-image pass plus one realistic
        full-page pass catch devices that only handle small shapes well
        (virtual/software adapters). When suspicious, alternate DirectML
        adapters are probed once and the fastest verified-DML engine kept.

        Returns elapsed milliseconds of the kept engine, or None on failure.
        """
        start = time.time()
        small_images = _build_warmup_images()
        stress_image = _build_stress_image()

        def measure(engine, imgs) -> float:
            t0 = time.time()
            for img in imgs:
                self._run(engine, img)
            return (time.time() - t0) * 1000

        try:
            small_ms = measure(self._engine, small_images)
            stress_ms = measure(self._engine, [stress_image])
        except Exception as e:
            logger.warning("Engine warmup failed (non-fatal): {}", e)
            return None

        if self._mode == "dml" and (
            small_ms > _WARMUP_SLOW_MS or stress_ms > _WARMUP_STRESS_SLOW_MS
        ):
            all_images = [*small_images, stress_image]
            stress_ms = self._probe_faster_dml_adapter(
                all_images, measure, small_ms, stress_ms
            )

        total = (time.time() - start) * 1000
        logger.info(
            "Engine pre-warmed in {:.0f} ms (small {:.0f} / full-page {:.0f}) [{}]",
            total,
            small_ms,
            stress_ms,
            self._label,
        )
        return total

    def _probe_faster_dml_adapter(self, images, measure, small_ms: float, current_ms: float) -> float:
        logger.warning(
            "DirectML health check failed (small {:.0f} ms, full-page {:.0f} ms); "
            "probing alternate adapters",
            small_ms,
            current_ms,
        )
        best_ms = current_ms
        best_params = dict(self._accel_params)
        best_engine = None

        for device_id in _DML_PROBE_DEVICE_IDS:
            params = {
                "EngineConfig.onnxruntime.use_dml": True,
                "EngineConfig.onnxruntime.dml_ep_cfg": {"device_id": device_id},
            }
            try:
                from rapidocr import RapidOCR

                _ensure_rapidocr_plain_dict_ep_cfg()
                candidate = RapidOCR(params=params)

                providers = _session_providers(candidate) or []
                if not providers or providers[0] != "DmlExecutionProvider":
                    logger.info(
                        "Adapter {}: not a usable DML device (providers={})",
                        device_id,
                        providers,
                    )
                    del candidate
                    continue

                ms = measure(candidate, images)
                logger.info(
                    "Adapter {} warmup: {:.0f} ms (current best {:.0f} ms)",
                    device_id,
                    ms,
                    best_ms,
                )
                if ms < best_ms * 0.8:  # only switch on a clear win
                    best_ms, best_params, best_engine = ms, params, candidate
            except Exception as e:
                logger.info("Adapter {} unusable: {}", device_id, e)

        if best_engine is not None:
            old, self._engine = self._engine, best_engine
            self._accel_params = best_params
            del old  # release losing sessions' VRAM
            import gc

            gc.collect()
            logger.warning(
                "Switched to faster DirectML adapter ({:.0f} ms vs {:.0f} ms)",
                best_ms,
                current_ms,
            )
        return best_ms

    # ------------------------------------------------------------- japanese

    @staticmethod
    def _looks_japanese(text: str) -> bool:
        kana = sum(
            1
            for ch in text
            if "\u3040" <= ch <= "\u30ff"  # hiragana + katakana
        )
        cjk_total = sum(
            1
            for ch in text
            if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff"
        )
        if cjk_total == 0:
            return False
        ratio = kana / cjk_total
        if ratio >= KANA_RATIO_THRESHOLD:
            logger.info(
                "Kana ratio {:.2f} >= {:.2f}, Japanese re-run considered",
                ratio,
                KANA_RATIO_THRESHOLD,
            )
        return ratio >= KANA_RATIO_THRESHOLD

    def _maybe_rerun_japanese(
        self, image: Image.Image, original: OcrResult
    ) -> OcrResult:
        if self._japan_model_path is None:
            logger.info(
                "Japanese-like text but no local japan model, keeping default result"
            )
            return original

        if self._japan_engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

            _ensure_rapidocr_plain_dict_ep_cfg()
            logger.info("Loading Japanese rec engine (offline local model)")
            # rapidocr requires Enum instances for ocr_version / model_type;
            # plain strings raise TypeError in ParseParams.update_batch.
            params = {
                "Rec.model_path": str(self._japan_model_path),
                "Rec.lang_type": LangRec.JAPAN,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Rec.model_type": ModelType.MOBILE,
            }
            # Run on the same accelerator as the main engine.
            params.update(self._accel_params)
            dict_file = japan_dict_path(self._japan_model_path)
            if dict_file is not None:
                params["Rec.rec_keys_path"] = str(dict_file)

            try:
                with _forbid_downloads():
                    self._japan_engine = RapidOCR(params=params)
            except Exception as e:
                logger.error("Japanese engine init failed: {}", e)
                self._japan_model_path = None
                return original

        try:
            jp = self._run(self._japan_engine, image)
        except Exception as e:
            logger.error("Japanese inference failed: {}", e)
            return original

        if not jp.lines:
            logger.info("Japanese pass empty, keeping default result")
            return original

        if jp.mean_score + 0.05 >= original.mean_score:
            logger.info(
                "Using Japanese result (score {:.3f} vs {:.3f})",
                jp.mean_score,
                original.mean_score,
            )
            return jp

        logger.info("Default result scored better, discarding Japanese pass")
        return original


def _build_warmup_images() -> list[Image.Image]:
    """
    Synthetic text screenshots covering det/cls/rec kernel shapes.

    Real text (not just bars) guarantees the detector finds regions, so
    recognition runs too. Two widths give recognition crops of different
    sizes, compiling more of the variable-shape kernels up front.
    """
    images: list[Image.Image] = []
    for width in (480, 800):
        img = Image.new("RGB", (width, 180), "white")
        draw = ImageDraw.Draw(img)
        y = 10
        for size in (16, 22, 28):
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1 fixed-size default only
                font = ImageFont.load_default()
            draw.text(
                (12, y),
                f"MaiOCR warmup {size} 0123456789 ABCabc",
                fill="black",
                font=font,
            )
            y += size + 16
        images.append(img)
    return images


def _build_stress_image() -> Image.Image:
    """
    A realistic full-page of text (many lines, varying widths).

    Exercises the large det input and heavy rec batching that tiny warmup
    images miss; weak/virtual adapters slow down disproportionately here.
    """
    img = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(img)
    y = 16
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    for i in range(34):
        width = 30 + (i * 37) % 90
        draw.text(
            (16, y),
            f"MaiOCR stress line {i:02d} " + "abcdefgh0123456789"[:width],
            fill="black",
            font=font,
        )
        y += 26
    return img

