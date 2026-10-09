import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maiocr.utils.logger import get_logger
from maiocr.utils.paths import external_models_dir

logger = get_logger()

# VRAM inactivity timeout in seconds (default 60s, configurable via env var)
VRAM_INACTIVITY_TIMEOUT = int(os.environ.get("MAIOCR_VRAM_TIMEOUT", "60"))

# Upper bound on detection input dimension to prevent multi-gigabyte VRAM blowups on 4K/multi-monitor
MAX_DET_SIDE_LEN = 2560

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
    Apply runtime optimizations and bug workarounds to RapidOCR:
    1. ProviderConfig.dml_ep_cfg: OmegaConf DictConfig rejection workaround.
    2. RapidOCR._initialize: avoid creating the direction classifier (text_cls)
       and its ONNX session when Global.use_cls is False, saving ~100 MiB VRAM
       and ~25% inference time.
    3. DetPreProcess.resize: upper-bound maximum detection input dimension to
       MAX_DET_SIDE_LEN (2560) to prevent multi-gigabyte VRAM blowups on 4K/multi-monitor.
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

    try:
        from rapidocr.main import RapidOCR

        orig_init = RapidOCR._initialize

        def optimized_initialize(self, cfg):
            orig_init(self, cfg)
            if not self.use_cls and hasattr(self, "text_cls") and self.text_cls is not None:
                # Discard unused classifier session immediately to avoid holding idle GPU VRAM
                cls_sess = getattr(self.text_cls, "session", None)
                if cls_sess is not None:
                    underlying = getattr(cls_sess, "session", None)
                    if underlying is not None:
                        try:
                            if hasattr(underlying, "__del__"):
                                underlying.__del__()
                        except Exception as e:
                            logger.debug("Error releasing classifier underlying session: {}", e)
                        cls_sess.session = None
                    self.text_cls.session = None
                self.text_cls = None

        RapidOCR._initialize = optimized_initialize
        logger.debug("Patched RapidOCR to avoid unused direction classifier session")
    except Exception as e:
        logger.warning("Could not patch RapidOCR classifier initialization: {}", e)

    try:
        import cv2
        from rapidocr.ch_ppocr_det.main import DetPreProcess
        from rapidocr.ch_ppocr_det.utils import ResizeImgError

        def capped_det_resize(self, img: np.ndarray):
            h, w = img.shape[:2]
            if self.limit_type == "max":
                ratio = (
                    float(self.limit_side_len) / max(h, w)
                    if max(h, w) > self.limit_side_len
                    else 1.0
                )
            else:
                ratio = (
                    float(self.limit_side_len) / min(h, w)
                    if min(h, w) < self.limit_side_len
                    else 1.0
                )

            # Guard against multi-gigabyte VRAM blowups on 4K/multi-monitor setups
            if max(h, w) * ratio > MAX_DET_SIDE_LEN:
                ratio = float(MAX_DET_SIDE_LEN) / max(h, w)

            resize_h = int(round(int(h * ratio) / 32) * 32)
            resize_w = int(round(int(w * ratio) / 32) * 32)
            if resize_w <= 0 or resize_h <= 0:
                return None
            try:
                return cv2.resize(img, (resize_w, resize_h))
            except Exception as exc:
                raise ResizeImgError from exc

        DetPreProcess.resize = capped_det_resize
        logger.debug(
            "Patched DetPreProcess with maximum dimension guard ({})",
            MAX_DET_SIDE_LEN,
        )
    except Exception as e:
        logger.warning("Could not patch DetPreProcess resize: {}", e)


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
    base = {
        "Global.text_score": _OCR_TEXT_SCORE,
        "Global.use_cls": False,
        "Rec.rec_batch_num": 4,
    }
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


class VRAMManager:
    """
    Tracks GPU activity and schedules VRAM release.

    The actual engine teardown is delegated to ``release_callback`` (owned
    by OCREngine) so state flags and reality can never diverge: whenever
    this manager says "VRAM not in use", the callback has already freed
    the GPU sessions.
    """

    def __init__(self, release_callback=None):
        self._vram_in_use = False
        self._last_gpu_activity = 0.0
        self._inactivity_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._release_callback = release_callback

    def mark_gpu_activity(self):
        """Call when GPU resources are about to be used (create/infer)."""
        with self._lock:
            self._last_gpu_activity = time.time()
            if not self._vram_in_use:
                self._vram_in_use = True
                logger.debug("VRAM marked as in use")
            self._reset_inactivity_timer()

    def _reset_inactivity_timer(self):
        if self._inactivity_timer:
            self._inactivity_timer.cancel()
        self._inactivity_timer = threading.Timer(
            VRAM_INACTIVITY_TIMEOUT, self._on_inactivity_timeout
        )
        self._inactivity_timer.daemon = True
        self._inactivity_timer.start()

    def _on_inactivity_timeout(self):
        with self._lock:
            idle_long_enough = (
                time.time() - self._last_gpu_activity >= VRAM_INACTIVITY_TIMEOUT
            )
            should_release = idle_long_enough and self._vram_in_use
        if not should_release:
            return
        if self._release_callback is None:
            logger.warning("VRAM idle timeout but no release callback set")
            return
        # Tear down outside the lock. The callback (OCREngine.release_vram)
        # clears the in-use flag only when it really frees the sessions;
        # if it refuses (e.g. inference busy), the flag stays True so the
        # displayed status remains truthful.
        released = bool(self._release_callback())
        if released:
            logger.info(
                "VRAM auto-released after {} s of GPU inactivity",
                VRAM_INACTIVITY_TIMEOUT,
            )
        else:
            # Still GPU-resident (busy or transient failure): re-arm so we
            # retry after another idle period instead of leaking the timer.
            with self._lock:
                self._reset_inactivity_timer()

    def force_release_vram(self) -> bool:
        """Clear the in-use flag; returns True if it was previously set."""
        with self._lock:
            if self._inactivity_timer:
                self._inactivity_timer.cancel()
                self._inactivity_timer = None
            was_in_use = self._vram_in_use
            self._last_gpu_activity = 0.0
            self._vram_in_use = False
            if was_in_use:
                logger.info("VRAM release requested")
            return was_in_use

    @property
    def vram_in_use(self) -> bool:
        with self._lock:
            return self._vram_in_use

    @property
    def status_text(self) -> str:
        """Human-readable VRAM status."""
        with self._lock:
            if self._vram_in_use:
                return "VRAM in use (GPU resident)"
            return "RAM resident (VRAM not in use)"


class OCREngine:
    """
    RapidOCR wrapper.

    - Uses the fastest available GPU provider (CUDA or DirectML) and falls
      back to CPU if GPU init/inference fails.
    - Optional Japanese pass when the result looks kana-heavy and a local
      japan rec model is present.
    - Manages VRAM: keeps model in RAM, loads to VRAM only during inference,
      auto-releases after 5 min inactivity.
    """

    def __init__(self):
        self._engine = None
        self._japan_engine = None
        self._japan_model_path: Path | None = find_japan_model()
        self._gpu_failures = 0
        # Guards engine create/destroy vs. concurrent release requests.
        self._lifecycle_lock = threading.RLock()
        # >0 while an inference is using the engines; blocks VRAM release.
        self._inference_active = 0

        self._mode, self._accel_params, self._label = detect_acceleration()
        # Engine creation is deferred to first inference (lazy): the app
        # stays RAM-resident at startup and only touches VRAM on demand.
        self._engine_initialized = False
        self._vram_manager = VRAMManager(release_callback=self.release_vram)

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
                with self._lifecycle_lock:
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
            with self._lifecycle_lock:
                self._engine = self._create_engine()
        except Exception as e:
            logger.error("CPU fallback also failed: {}", e)
            raise

    # ------------------------------------------------------------ inference

    def recognize(self, image: Image.Image) -> OcrResult:
        # Mark GPU activity first: VRAM status must be truthful from the
        # moment resources are allocated (engine create + warmup included).
        self._vram_manager.mark_gpu_activity()
        self.ensure_gpu_engines()
        try:
            with self._lifecycle_lock:
                self._inference_active += 1
            # Direct inference compiles exact kernels for the user's capture shape.
            # Bypassing synthetic warmup cuts ~700ms from the first-capture latency
            # while avoiding intermediate allocation of throwaway shapes.
            self._warmup_done = True

            result = self._run(self._engine, image)
            self._gpu_failures = 0  # healthy run resets the recovery budget

            if self._looks_japanese(result.text):
                result = self._maybe_rerun_japanese(image, result)
        finally:
            with self._lifecycle_lock:
                self._inference_active -= 1
            self._vram_manager.mark_gpu_activity()

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

    # ------------------------------------------------------------ VRAM management

    def release_vram(self) -> bool:
        """
        Tear down GPU-resident sessions and return to RAM-resident state.

        Refuses (returns False) while an inference is in flight - freeing
        sessions under a running call is what wedged/crashed the DML
        device previously. Also returns False on CPU-only setups or when
        VRAM is not currently held.
        """
        if not self._accel_params:
            return False

        with self._lifecycle_lock:
            if self._inference_active > 0:
                logger.info(
                    "VRAM release skipped: inference in progress"
                )
                return False

            was_in_use = self._vram_manager.force_release_vram()
            if self._engine is not None or self._japan_engine is not None:
                self._release_gpu_engines()
            return was_in_use

    def _release_gpu_engines(self):
        """Release GPU engine objects so the driver can reclaim VRAM."""
        try:
            for name in ("_engine", "_japan_engine"):
                engine = getattr(self, name, None)
                if engine is None:
                    continue
                # Drop internal ONNX Runtime session references first so
                # the native memory is freed even if the RapidOCR wrapper
                # itself lingers in a reference cycle until gc runs.
                for attr in ("text_det", "text_rec", "text_cls"):
                    session_obj = getattr(engine, attr, None)
                    if session_obj is None:
                        continue
                    wrapper = getattr(session_obj, "session", None)
                    if wrapper is not None:
                        underlying = getattr(wrapper, "session", None)
                        if underlying is not None:
                            try:
                                if hasattr(underlying, "__del__"):
                                    underlying.__del__()
                            except Exception as e:
                                logger.debug("Error deleting underlying session: {}", e)
                            wrapper.session = None
                        try:
                            if hasattr(wrapper, "__del__"):
                                wrapper.__del__()
                        except Exception as e:
                            logger.debug("Error deleting session wrapper: {}", e)
                        session_obj.session = None
                    setattr(engine, attr, None)
                setattr(self, name, None)

            self._engine_initialized = False
            import gc
            gc.collect()

            logger.info("GPU engines released, VRAM freed")
        except Exception as e:
            logger.warning("Error releasing GPU engines: {}", e)

    def ensure_gpu_engines(self):
        """Create the engine on first use, or re-create after a VRAM release."""
        with self._lifecycle_lock:
            if not self._engine_initialized or self._engine is None:
                logger.info("Loading OCR engine into GPU memory on demand")
                self._engine = self._create_engine()
                self._engine_initialized = True

    @property
    def vram_status(self) -> str:
        """Current VRAM status text for display."""
        if not self._accel_params:
            return "CPU 模式 (无 GPU 加速)"
        return self._vram_manager.status_text

    @property
    def vram_in_use(self) -> bool:
        """Whether VRAM is currently allocated for GPU inference."""
        if not self._accel_params:
            return False
        return self._vram_manager.vram_in_use

    def warmup(self, probe_adapters: bool = False) -> float | None:
        """
        Run throwaway inferences so GPU kernels compile and buffers
        allocate before the first real capture.

        DirectML/CUDA compile kernels lazily per input shape; without this
        the first user-facing OCR pays a multi-second compilation cost.

        With ``probe_adapters`` (off by default) a full-page stress pass
        and the alternate-adapter health check run as well; that variant
        is only meant for explicit startup-style warmups.

        Returns elapsed milliseconds of the kept engine, or None on failure.
        """
        start = time.time()
        small_images = _build_warmup_images()

        def measure(engine, imgs) -> float:
            t0 = time.time()
            for img in imgs:
                self._run(engine, img)
            return (time.time() - t0) * 1000

        try:
            small_ms = measure(self._engine, small_images)

            if probe_adapters:
                # Full health check: realistic full-page pass plus, on
                # suspiciously slow adapters, a one-time probe of alternate
                # DirectML devices. Only safe at startup-style moments;
                # skipped on the lazy first-use path because the rapid
                # session churn can destabilise the DML device mid-session
                # and hang subsequent inferences.
                stress_image = _build_stress_image()
                stress_ms = measure(self._engine, [stress_image])
                if self._mode == "dml" and (
                    small_ms > _WARMUP_SLOW_MS
                    or stress_ms > _WARMUP_STRESS_SLOW_MS
                ):
                    all_images = [*small_images, stress_image]
                    self._probe_faster_dml_adapter(
                        all_images, measure, small_ms, stress_ms
                    )
        except Exception as e:
            logger.warning("Engine warmup failed (non-fatal): {}", e)
            return None

        total = (time.time() - start) * 1000
        logger.info(
            "Engine pre-warmed in {:.0f} ms (small {:.0f}) [{}]",
            total,
            small_ms,
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

