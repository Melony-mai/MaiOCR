from PySide6.QtCore import QObject, Signal


class HotkeyManager(QObject):
    """
    Platform hotkey listener.

    Handlers map:
        "text": callable  -> regular OCR
        "code": callable  -> code OCR

    Registration/runtime failures are reported through the
    `error_occurred` Signal so the UI can notify the user (the signal is
    emitted from the listener thread and delivered via queued connection).
    Once listening, `bound` reports the shortcuts actually registered
    (they may differ from the defaults when the OS owns one of them).
    """

    error_occurred = Signal(str)
    bound = Signal(dict)

    def __init__(self):
        super().__init__()
        self._bound_info: dict = {}

    @property
    def bound_info(self) -> dict:
        """Shortcut descriptions actually registered, e.g. {"text": "Win+PrtSc"}."""
        return self._bound_info

    def _set_bound(self, info: dict):
        self._bound_info = dict(info)
        self.bound.emit(self._bound_info)

    def start(self, handlers: dict):
        """Start listening. Blocking work must run on its own thread."""
        raise NotImplementedError

    def stop(self):
        """Stop listening and release resources."""
        raise NotImplementedError
