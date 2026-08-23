"""Windows global hotkeys via RegisterHotKey (no extra dependencies).

    Win + PrtSc        -> regular OCR
    Win + Ctrl + PrtSc -> code OCR

On Windows 10/11, Explorer pre-registers OS screenshot combos (including
Win+PrtSc) at startup, so RegisterHotKey fails there with error 1409 even
with no third-party software running. Each binding therefore has a
fallback candidate (e.g. Win+Shift+PrtSc) and is registered
independently: a taken combo degrades to its fallback instead of killing
all hotkeys. The UI is told what was actually bound via the `bound`
signal.
"""

import ctypes
import threading
import time
from ctypes import wintypes

from maiocr.hotkey.base import HotkeyManager
from maiocr.utils.logger import get_logger

logger = get_logger()

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_QUIT = 0x0012
WM_HOTKEY = 0x0312

ERROR_HOTKEY_ALREADY_REGISTERED = 1409

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

VK_SNAPSHOT = 0x2C  # PrintScreen

HOTKEY_TEXT = 1
HOTKEY_CODE = 2

# (binding key, [(modifiers, label), ...]) - first entry is the primary
# combo, later entries are fallbacks tried in order.
_BINDINGS = [
    (
        "text",
        [
            (MOD_WIN, "Win+PrtSc"),
            (MOD_SHIFT | MOD_WIN, "Win+Shift+PrtSc"),
        ],
    ),
    (
        "code",
        [
            (MOD_CONTROL | MOD_WIN, "Win+Ctrl+PrtSc"),
            (MOD_CONTROL | MOD_SHIFT | MOD_WIN, "Win+Ctrl+Shift+PrtSc"),
        ],
    ),
]

_REGISTER_ATTEMPTS = 5
_REGISTER_RETRY_DELAY_S = 0.3


_MAX_RESTARTS = 5
_RESTART_DELAY_S = 3.0


class WindowsHotkeyManager(HotkeyManager):
    """
    RegisterHotKey listener running its message loop off the UI thread.

    Self-healing: if the listener crashes or loses registration (driver
    hiccups, explorer restarts, conflicts appearing mid-session), it is
    restarted automatically with backoff; the user is notified only when
    all attempts are exhausted.
    """

    def __init__(self):
        super().__init__()
        self._handlers: dict = {}
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._registered: list[int] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._restart_budget = _MAX_RESTARTS

    # ------------------------------------------------------------------ API

    def start(self, handlers: dict):
        self._handlers = dict(handlers or {})
        self._stop_event.clear()
        self._restart_budget = _MAX_RESTARTS
        self._spawn()

    def _spawn(self):
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="maiocr-hotkey",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        logger.info("Stopping hotkey listener")
        self._stop_event.set()
        try:
            tid = self._thread_id
            # PostThreadMessage lives in user32, not kernel32. WM_QUIT is
            # what wakes the GetMessageW loop and lets it unregister the
            # hotkeys; failing to deliver it used to crash the quit path
            # and leave stale processes holding the shortcuts.
            if (
                tid is not None
                and not _user32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
            ):
                logger.error(
                    "Failed to post WM_QUIT to hotkey thread {}: error={}",
                    tid,
                    ctypes.get_last_error(),
                )
        except Exception as e:
            # Never let shutdown blow up here; the join below still runs.
            logger.exception("Stopping hotkey listener failed: {}", e)

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None
        self._thread_id = None

    # ------------------------------------------------------------- internals

    def _try_register(self, hotkey_id: int, modifiers: int, vk: int) -> int:
        """
        Register one hotkey, retrying while the combo is still held by a
        previous instance that is shutting down.

        Returns 0 on success, otherwise the last Win32 error code.
        """
        error = 0
        for attempt in range(_REGISTER_ATTEMPTS):
            if _user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                return 0
            error = ctypes.get_last_error()
            if error != ERROR_HOTKEY_ALREADY_REGISTERED:
                return error
            if attempt < _REGISTER_ATTEMPTS - 1:
                time.sleep(_REGISTER_RETRY_DELAY_S)
        return error

    def _register_all(self) -> tuple[list[int], dict]:
        """
        Register every binding independently, falling back per binding.

        Returns (registered ids, bound info dict). Raises RuntimeError
        only when a binding has no usable combo at all.
        """
        ids_for_key = {"text": HOTKEY_TEXT, "code": HOTKEY_CODE}
        registered: list[int] = []
        info: dict = {}
        problems: list[str] = []

        for key, candidates in _BINDINGS:
            bound_label = None
            for index, (modifiers, label) in enumerate(candidates):
                error = self._try_register(
                    ids_for_key[key], modifiers | MOD_NOREPEAT, VK_SNAPSHOT
                )
                if error == 0:
                    registered.append(ids_for_key[key])
                    bound_label = label
                    logger.info(
                        "Hotkey registered: {} -> {} (id={} mods=0x{:X} vk=0x{:X})",
                        key,
                        label,
                        ids_for_key[key],
                        modifiers,
                        VK_SNAPSHOT,
                    )
                    info[f"{key}_fallback"] = index > 0
                    break
                if index == 0:
                    logger.warning(
                        "Primary hotkey {} unavailable (error={}); trying fallback",
                        label,
                        error,
                    )
            if bound_label is not None:
                info[key] = bound_label
            else:
                primary = candidates[0][1]
                logger.error("No usable combo for {} binding: {}", key, primary)
                problems.append(
                    f"{primary} 及其备选组合均注册失败"
                )

        if problems:
            for done_id in registered:
                _user32.UnregisterHotKey(None, done_id)
            raise RuntimeError(
                "注册全局热键失败：" + "；".join(problems)
                + "，快捷键可能被系统或其他程序占用"
            )
        return registered, info

    def _run_guarded(self):
        """
        Supervise the listener thread: restart on crashes or registration
        failures with a fixed backoff until the restart budget is spent.
        A clean stop (WM_QUIT) exits immediately.
        """
        while True:
            last_error: Exception | None = None
            try:
                self._run()
            except Exception as e:
                last_error = e
                logger.error("Hotkey listener failed: {}", e)

            if self._stop_event.is_set():
                break

            if last_error is None:
                break  # clean exit via WM_QUIT

            if self._restart_budget <= 0:
                # Single user-facing notification, only when all is lost.
                self.error_occurred.emit(
                    f"全局热键监听已停止（多次重试失败）：{last_error}"
                )
                break

            self._restart_budget -= 1
            logger.warning(
                "Hotkey listener down ({}); restarting in {}s ({} attempts left)",
                last_error,
                _RESTART_DELAY_S,
                self._restart_budget,
            )
            if self._stop_event.wait(_RESTART_DELAY_S):
                break

    def _run(self):
        self._thread_id = _kernel32.GetCurrentThreadId()

        # Raises RuntimeError when no combo can be registered; the guard
        # above decides whether to retry or give up.
        with self._lock:
            self._registered, info = self._register_all()
        if not self._registered:
            return

        # Healthy registration refills the restart budget.
        self._restart_budget = _MAX_RESTARTS
        self._set_bound(info)

        try:
            msg = wintypes.MSG()
            while True:
                ret = _user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:  # WM_QUIT or error
                    break
                if msg.message == WM_HOTKEY:
                    self._dispatch(int(msg.wParam))
        finally:
            with self._lock:
                for hotkey_id in self._registered:
                    _user32.UnregisterHotKey(None, hotkey_id)
                self._registered = []
            logger.info("Hotkey listener stopped")

    def _dispatch(self, hotkey_id: int):
        key = "text" if hotkey_id == HOTKEY_TEXT else "code"
        callback = self._handlers.get(key)
        if callback is None:
            logger.warning("No handler bound for hotkey id={}", hotkey_id)
            return

        logger.info("Hotkey triggered: {}", key)
        threading.Thread(
            target=self._safe_call,
            args=(callback,),
            name="maiocr-trigger",
            daemon=True,
        ).start()

    @staticmethod
    def _safe_call(callback):
        try:
            callback()
        except Exception as e:
            logger.exception("Hotkey handler failed: {}", e)
