"""Unit tests for the Windows hotkey registration ladder (no real hooks)."""

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only hotkey internals"
)


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def _manager():
    from maiocr.hotkey.windows import WindowsHotkeyManager

    return WindowsHotkeyManager()


def test_primary_combos_register_cleanly():
    from maiocr.hotkey.windows import HOTKEY_CODE, HOTKEY_TEXT

    m = _manager()
    seen = []

    def fake_try(hotkey_id, modifiers, vk):
        seen.append((hotkey_id, modifiers))
        return 0

    m._try_register = fake_try
    ids, info = m._register_all()

    assert ids == [HOTKEY_TEXT, HOTKEY_CODE]
    assert info == {
        "text": "Win+PrtSc",
        "code": "Win+Ctrl+PrtSc",
        "text_fallback": False,
        "code_fallback": False,
    }
    # NOREPEAT must always be part of the registration.
    assert all(mods & 0x4000 for _, mods in seen)


def test_os_owned_combo_falls_back_per_binding():
    """Explorer owns Win+PrtSc on Win10/11; text mode must degrade alone."""
    from maiocr.hotkey.windows import (
        HOTKEY_CODE,
        HOTKEY_TEXT,
        MOD_SHIFT,
        MOD_WIN,
    )

    m = _manager()

    def fake_try(hotkey_id, modifiers, vk):
        if hotkey_id == HOTKEY_TEXT and modifiers & MOD_WIN and not modifiers & MOD_SHIFT:
            return 1409  # primary taken by the OS
        return 0

    m._try_register = fake_try
    ids, info = m._register_all()

    assert ids == [HOTKEY_TEXT, HOTKEY_CODE]
    assert info["text"] == "Win+Shift+PrtSc"
    assert info["text_fallback"] is True
    assert info["code"] == "Win+Ctrl+PrtSc"
    assert info["code_fallback"] is False


def test_binding_without_any_combo_is_fatal():
    from maiocr.hotkey.windows import HOTKEY_TEXT

    m = _manager()
    m._try_register = lambda hotkey_id, modifiers, vk: (
        1409 if hotkey_id == HOTKEY_TEXT else 0
    )

    with pytest.raises(RuntimeError):
        m._register_all()


def test_bound_signal_carries_effective_bindings():
    from maiocr.hotkey.windows import HOTKEY_TEXT, MOD_SHIFT

    m = _manager()
    received = []
    m.bound.connect(received.append)

    def fake_try(hotkey_id, modifiers, vk):
        primary = hotkey_id == HOTKEY_TEXT and not modifiers & MOD_SHIFT
        return 1409 if primary else 0

    m._try_register = fake_try
    ids, info = m._register_all()
    m._set_bound(info)  # same call _run() makes after successful registration

    assert received and received[0]["text"].startswith("Win+Shift")
    assert m.bound_info == received[0]
    assert ids


def test_stop_is_safe_before_start():
    m = _manager()
    m.stop()  # must not raise even though no thread exists


def test_listener_restarts_after_registration_failures(monkeypatch, qapp):
    """Crashed/failed listeners must self-heal without user action."""
    import time

    from maiocr.hotkey import windows as win
    from maiocr.hotkey.windows import WindowsHotkeyManager

    monkeypatch.setattr(win, "_RESTART_DELAY_S", 0.05)

    m = WindowsHotkeyManager()
    bound = []
    errors = []
    m.bound.connect(bound.append)
    m.error_occurred.connect(errors.append)

    attempts = {"n": 0}

    def flaky_register():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient conflict")
        return [win.HOTKEY_TEXT], {
            "text": "Win+PrtSc",
            "text_fallback": False,
        }

    monkeypatch.setattr(m, "_register_all", flaky_register)
    m.start({})

    deadline = time.time() + 5
    while not bound and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert bound, "listener never recovered"
    assert attempts["n"] == 3
    assert not errors  # no user-facing error for transient failures
    assert m._restart_budget == win._MAX_RESTARTS  # refilled on success
    m.stop()
    qapp.processEvents()


def test_listener_gives_up_after_budget_and_notifies_once(monkeypatch, qapp):
    import time

    from maiocr.hotkey import windows as win
    from maiocr.hotkey.windows import WindowsHotkeyManager

    monkeypatch.setattr(win, "_RESTART_DELAY_S", 0.05)

    m = WindowsHotkeyManager()
    errors = []
    m.error_occurred.connect(errors.append)

    def always_fail():
        raise RuntimeError("combo taken")

    monkeypatch.setattr(m, "_register_all", always_fail)
    m.start({})

    deadline = time.time() + 10
    while len(errors) < 1 and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.05)

    assert len(errors) == 1  # exactly one notification at final give-up
    assert "combo taken" in errors[0]
    m.stop()
    qapp.processEvents()


def test_stop_during_backoff_exits_promptly(monkeypatch):
    import threading
    import time

    from maiocr.hotkey import windows as win
    from maiocr.hotkey.windows import WindowsHotkeyManager

    monkeypatch.setattr(win, "_RESTART_DELAY_S", 30.0)

    m = WindowsHotkeyManager()

    def always_fail():
        raise RuntimeError("down")

    monkeypatch.setattr(m, "_register_all", always_fail)
    started = threading.Event()

    orig_spawn = m._spawn
    seen = {}

    def spawn():
        orig_spawn()
        seen["t"] = m._thread
        started.set()

    monkeypatch.setattr(m, "_spawn", spawn)
    m.start({})
    assert started.wait(2)
    t0 = time.time()
    m.stop()
    assert time.time() - t0 < 5  # backoff wait must be interruptible
    assert not seen["t"].is_alive()
