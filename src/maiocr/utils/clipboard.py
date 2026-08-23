from maiocr.utils.logger import get_logger

logger = get_logger()


def _copy_qt(text: str) -> bool:
    """
    Copy via the Qt clipboard.

    Preferred whenever a QApplication exists (the GUI always has one):
    native Win32 clipboard on Windows, no helper subprocesses. Avoids
    pyperclip's wl-copy/xclip daemons, which fork and hold inherited
    pipes open - hanging any caller that reads our output through a pipe.
    """
    try:
        from PySide6.QtGui import QGuiApplication

        if QGuiApplication.instance() is None:
            return False

        clipboard = QGuiApplication.clipboard()
        clipboard.setText(text)
        return True
    except Exception as e:
        logger.warning("Qt clipboard failed: {}", e)
        return False


def _copy_pyperclip(text: str) -> bool:
    """
    Headless fallback (CLI tools / tests without a display).

    pyperclip shells out to xclip/wl-copy, whose helper processes fork and
    stay resident while inheriting our stdio. Redirect stdio to devnull at
    the fd level so they can never hold our pipes open.
    """
    import os

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.warning("pyperclip failed: {}", e)
        return False
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def copy_text(text: str) -> bool:
    """
    Copy text to the system clipboard.

    Returns False for empty/whitespace-only input; tries Qt first,
    then pyperclip.
    """
    if not text or not text.strip():
        logger.warning("Clipboard skipped: empty text")
        return False

    copied = _copy_qt(text) or _copy_pyperclip(text)

    if copied:
        logger.info("Copied text to clipboard: {} chars", len(text))
    else:
        logger.error("All clipboard backends failed")
    return copied
