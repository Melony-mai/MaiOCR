"""Windows process-level tuning so OCR never gets scheduled like a background chore.

A tray application can be tagged by Windows with EcoQoS (power throttling),
which parks it on efficiency cores; the OCR pipeline interleaves heavy CPU
pre/post-processing between DirectML dispatches, so a starved CPU makes the
whole pipeline crawl while the GPU sits idle. Raising priority and opting
out of EcoQoS keeps the feed to the GPU full.
"""

import ctypes
import sys

from maiocr.utils.logger import get_logger

logger = get_logger()

_ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
_ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS enum
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1

_DONE = False


def tune_process_for_interactivity() -> None:
    """Raise priority and disable EcoQoS. Best-effort, Windows-only."""
    global _DONE
    if _DONE or sys.platform != "win32":
        return
    _DONE = True

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetCurrentProcess()

        # Above-normal priority: OCR bursts should preempt ordinary work,
        # but stay below realtime/machine-critical classes.
        if not kernel32.SetPriorityClass(handle, _ABOVE_NORMAL_PRIORITY_CLASS):
            logger.debug(
                "SetPriorityClass failed: error {}", ctypes.get_last_error()
            )

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        # ControlMask selects execution-speed throttling; StateMask=0 opts out.
        state = PowerThrottlingState(
            Version=1,
            ControlMask=_PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask=0,
        )
        if not kernel32.SetProcessInformation(
            handle,
            _ProcessPowerThrottling,
            ctypes.byref(state),
            ctypes.sizeof(state),
        ):
            logger.debug(
                "SetProcessInformation(EcoQoS opt-out) failed: error {}",
                ctypes.get_last_error(),
            )

        logger.info("Process tuned: above-normal priority, EcoQoS disabled")
    except Exception as e:
        logger.debug("Process tuning unavailable: {}", e)
