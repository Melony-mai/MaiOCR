import sys
import threading

from loguru import logger

from maiocr.utils.paths import logs_dir


def _setup_sink():
    logger.remove()
    if not getattr(sys, "frozen", False):
        logger.add(
            sys.stderr,
            level="INFO",
            backtrace=False,
        )
    logger.add(
        logs_dir() / "maiocr.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


_setup_sink()

_installed = False


def install_exception_hooks():
    """Log uncaught exceptions in the main thread and any thread."""
    global _installed
    if _installed:
        return
    _installed = True

    def _sys_hook(exc_type, exc_value, exc_tb):
        logger.opt(exception=(exc_type, exc_value, exc_tb)).error(
            "Uncaught exception: {}", exc_value
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        logger.opt(exception=args.exc_type).error(
            "Uncaught exception in thread {}: {}",
            args.thread.name if args.thread else "?",
            args.exc_value,
        )

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


def get_logger():
    return logger
