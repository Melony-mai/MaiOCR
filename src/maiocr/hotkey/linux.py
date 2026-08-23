import glob
import threading

from evdev import InputDevice, ecodes

from maiocr.hotkey.base import HotkeyManager
from maiocr.utils.logger import get_logger

logger = get_logger()


SHIFT_KEYS = ("KEY_LEFTSHIFT", "KEY_RIGHTSHIFT")

TRIGGER_KEYS = ("KEY_ENTER", "KEY_KPENTER")


def find_keyboard() -> str | None:
    """
    Auto detect a keyboard device under /dev/input.
    """

    candidates = []

    for path in sorted(glob.glob("/dev/input/event*")):

        try:
            device = InputDevice(path)
        except (PermissionError, OSError):
            continue


        caps = device.capabilities(verbose=False)

        if ecodes.EV_KEY not in caps:
            continue


        keys = set(caps[ecodes.EV_KEY])


        has_letters = {ecodes.KEY_A, ecodes.KEY_Q} <= keys

        is_mouse = ecodes.BTN_MOUSE in keys or ecodes.BTN_TOUCH in keys


        if has_letters and not is_mouse:
            logger.info(
                f"Found keyboard: {path} ({device.name})"
            )

            return path


        candidates.append(path)


    if candidates:
        logger.info(
            f"Using fallback input device: {candidates[0]}"
        )

        return candidates[0]


    return None


class LinuxHotkeyManager(HotkeyManager):

    def __init__(self, device_path=None, hotkey="shift+enter"):
        super().__init__()

        self.device_path = (
            device_path
            or find_keyboard()
        )


        if self.device_path is None:
            raise RuntimeError(
                "No keyboard device found under /dev/input"
            )


        self.device = InputDevice(self.device_path)

        self.running = False

        self.thread = None

        self.callback = None


    def start(self, handlers: dict):
        callback = (handlers or {}).get("text")
        if handlers and "code" in handlers:
            logger.warning(
                "Linux dev hotkeys support text mode only; code hotkey ignored"
            )

        self.callback = callback

        self.running = True


        # 避免按键事件同时转发到桌面环境
        try:
            self.device.grab()
        except OSError as e:
            logger.warning(
                f"Cannot grab device: {e}"
            )


        self.thread = threading.Thread(
            target=self._listen,
            name="maiocr-hotkey",
            daemon=True,
        )

        self.thread.start()


        logger.info(
            f"Hotkey listener started: Shift+Enter on {self.device_path}"
        )


    def _listen(self):

        shift_pressed = False


        try:

            for event in self.device.read_loop():

                if not self.running:
                    break


                if event.type != ecodes.EV_KEY:
                    continue


                key_event = ecodes.KEY.get(
                    event.code,
                    event.code
                )


                # 调试按键
                logger.debug(
                    f"KEY {key_event} value={event.value}"
                )


                # 按下
                if event.value == 1:

                    if key_event in SHIFT_KEYS:
                        shift_pressed = True


                    elif (
                        shift_pressed
                        and key_event in TRIGGER_KEYS
                    ):

                        logger.info(
                            "Hotkey triggered"
                        )

                        self._fire()


                # 松开
                elif event.value == 0 and key_event in SHIFT_KEYS:
                    shift_pressed = False


        except (OSError, ValueError):
            logger.info(
                "Hotkey listener stopped"
            )


    def _fire(self):

        if self.callback is None:
            return


        thread = threading.Thread(
            target=self.callback,
            daemon=True,
        )

        thread.start()


    def stop(self):

        logger.info(
            "Stopping hotkey listener"
        )

        self.running = False


        try:
            self.device.ungrab()
        except OSError:
            pass


        try:
            self.device.close()
        except OSError:
            pass
