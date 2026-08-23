import sys
import threading
import time

from PySide6.QtCore import QMetaObject, QObject, Qt, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from maiocr.core.pipeline import MaiOCRPipeline, Mode, PipelineResult
from maiocr.hotkey.manager import create_hotkey_manager
from maiocr.ui.result import ResultWindow
from maiocr.ui.selector import RegionSelector
from maiocr.ui.tray import TrayIcon
from maiocr.utils.clipboard import copy_text
from maiocr.utils.logger import get_logger, install_exception_hooks
from maiocr.utils.paths import icon_path
from maiocr.utils.performance import tune_process_for_interactivity
from maiocr.utils.singleinstance import SingleInstance

logger = get_logger()


class OCRController(QObject):
    """
    Bridges hotkey/tray events to the OCR pipeline and the UI thread.

    Every trigger - hotkey callbacks arrive on non-GUI threads - is routed
    through `region_requested` so the selection overlay always opens on the
    Qt main thread. Pipeline work then runs on worker threads; results come
    back via queued signal connections. The automatic clipboard copy happens
    in a main-thread slot because QClipboard must only be touched from the
    GUI thread.
    """

    result_ready = Signal(object)      # PipelineResult
    error_occurred = Signal(str)
    notice = Signal(str)
    region_requested = Signal(object)  # Mode

    def __init__(self):
        super().__init__()
        self._pipeline: MaiOCRPipeline | None = None
        self._busy_lock = threading.Lock()
        self._selecting = False
        self.region_requested.connect(self._on_region_requested)

    def start(self):
        """Load models in the background so the tray shows up immediately."""
        threading.Thread(
            target=self._init_pipeline,
            name="maiocr-init",
            daemon=True,
        ).start()

    def _init_pipeline(self):
        try:
            pipeline = MaiOCRPipeline()
            self._pipeline = pipeline
            logger.info("OCR pipeline initialized")
            # Pre-warm included: the very first capture is already fast.
            self.notice.emit(f"OCR 引擎已就绪（{pipeline.acceleration}）")
        except Exception as e:
            logger.exception("Pipeline initialization failed: {}", e)
            self.error_occurred.emit(f"OCR 引擎初始化失败：{e}")

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------- triggers

    # Safe to call from any thread: the signal is delivered (directly or via
    # queued connection) on the GUI thread.
    def trigger_text(self):
        self.region_requested.emit(Mode.TEXT)

    def trigger_code(self):
        self.region_requested.emit(Mode.CODE)

    # ------------------------------------------------------------ internals

    @Slot(object)
    def _on_region_requested(self, mode: Mode):
        if self._selecting:
            logger.info("Region selection already active; trigger ignored")
            return
        if not self.ready:
            logger.warning("Trigger ignored: pipeline still loading")
            self.notice.emit("模型加载中，请稍候…")
            return

        self._selecting = True
        try:
            region = RegionSelector.select_region()
        finally:
            self._selecting = False

        if region is None:
            logger.info("Region capture cancelled")
            return
        self._spawn(mode, region=region)

    def _spawn(self, mode: Mode, image=None, region=None):
        if not self._busy_lock.acquire(blocking=False):
            logger.warning("OCR task skipped: previous task still running")
            self.notice.emit("正在处理上一次识别，请稍候")
            return

        worker_kwargs = {"mode": mode, "image": image, "region": region}
        threading.Thread(
            target=self._run,
            kwargs=worker_kwargs,
            name="maiocr-ocr",
            daemon=True,
        ).start()

    def _run(self, mode: Mode, image, region):
        try:
            result: PipelineResult = self._pipeline.run(
                mode=mode, image=image, region=region
            )
            self.result_ready.emit(result)
            if not result.output.strip():
                self.notice.emit("未识别到内容，请调整截图范围后重试")
        except Exception as e:
            logger.exception("OCR task failed: {}", e)
            self.error_occurred.emit(f"识别失败：{e}")
        finally:
            self._busy_lock.release()


def main() -> int:
    install_exception_hooks()
    tune_process_for_interactivity()

    QApplication.setApplicationName("MaiOCR")
    QApplication.setOrganizationName("MaiOCR")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    icon_file = icon_path()
    if icon_file is not None:
        app.setWindowIcon(QIcon(str(icon_file)))

    guard = SingleInstance()
    if not guard.acquire():
        QMessageBox.information(
            None,
            "MaiOCR",
            "MaiOCR 已在运行中，请查看系统托盘图标。",
        )
        return 0

    controller = OCRController()

    result_window = ResultWindow()

    tray = TrayIcon(
        on_text=controller.trigger_text,
        on_code=controller.trigger_code,
        on_show_history=result_window.show_history,
        on_quit=app.quit,
    )

    # Copy first (main thread), then render - add_result displays whether
    # the output was copied.
    def _on_result_ready(result: PipelineResult):
        result.copied = (
            bool(result.output.strip()) and copy_text(result.output)
        )

    controller.result_ready.connect(_on_result_ready)
    controller.result_ready.connect(result_window.add_result)

    def _on_error(message: str):
        logger.error("User-facing error: {}", message)
        tray.notify(message, error=True)

    def _on_notice(message: str):
        tray.notify(message)

    # Connected before hotkey.start() so the (thread-emitted) bound signal
    # can never slip through unhandled.
    def _on_hotkeys_bound(info: dict):
        text_label = info.get("text", "Win+PrtSc")
        code_label = info.get("code", "Win+Ctrl+PrtSc")
        tray.update_shortcut_labels(text_label, code_label)

        notes = []
        if info.get("text_fallback"):
            notes.append(
                f"Win+PrtSc 已被系统或其他程序占用，截图识别文字改用 {text_label}"
            )
        if info.get("code_fallback"):
            notes.append(
                f"Win+Ctrl+PrtSc 已被系统或其他程序占用，截图识别代码改用 {code_label}"
            )
        if notes:
            logger.warning("Hotkey fallback in effect: {}", info)
            tray.notify("\n".join(notes))

    controller.error_occurred.connect(_on_error)
    controller.notice.connect(_on_notice)

    hotkey = create_hotkey_manager()
    hotkey.error_occurred.connect(_on_error)
    hotkey.bound.connect(_on_hotkeys_bound)
    try:
        hotkey.start(
            {
                "text": controller.trigger_text,
                "code": controller.trigger_code,
            }
        )
    except Exception as e:
        logger.exception("Hotkey registration failed: {}", e)
        _on_error(str(e))

    app.aboutToQuit.connect(hotkey.stop)
    app.aboutToQuit.connect(guard.release)

    # Start model loading only after every signal is connected, so early
    # failures can never be emitted into the void.
    controller.start()

    tray.show()
    tray.notify("MaiOCR 已启动，全局快捷键注册后即可框选截图识别")
    logger.info("MaiOCR started")

    if "--bench" in sys.argv:
        _run_bench(controller, app)

    return app.exec()


def _run_bench(controller: "OCRController", app: QApplication):
    """
    Diagnostic mode (`MaiOCR.exe --bench`): wait for readiness, then run
    timed full-screen OCR passes and log them. Used to compare packaged
    vs development environments; exits the app afterwards.
    """

    def worker():
        deadline = time.time() + 90
        while not controller.ready and time.time() < deadline:
            time.sleep(0.25)
        if not controller.ready:
            logger.error("BENCH: pipeline never became ready")
            QMetaObject.invokeMethod(app, "quit", Qt.QueuedConnection)
            return
        for i in range(3):
            result = controller._pipeline.run(mode=Mode.TEXT)
            logger.info(
                "BENCH pass{} ocr={:.0f}ms total={:.0f}ms chars={}",
                i + 1,
                result.ocr_ms,
                result.total_ms,
                len(result.output),
            )
        QMetaObject.invokeMethod(app, "quit", Qt.QueuedConnection)

    threading.Thread(target=worker, name="maiocr-bench", daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
