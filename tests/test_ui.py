"""Offscreen smoke tests for Qt widgets."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_result_window_add_and_browse(qapp):
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import ResultWindow

    win = ResultWindow()

    for i in range(3):
        win.add_result(
            PipelineResult(
                mode=Mode.CODE if i % 2 else Mode.TEXT,
                raw_text=f"raw {i}",
                output=f"```python\ncode {i}\n```" if i % 2 else f"text {i}",
                language="python" if i % 2 else "",
                is_code=bool(i % 2),
            )
        )

    assert win.list_widget.count() == 3
    assert "text 2" in win.text_edit.toPlainText()
    assert win._records[0].output == "text 2"
    assert win._records[1].output.startswith("```python")
    assert win._records[2].raw_text == "raw 0"

    win.clear_history()
    assert win.list_widget.count() == 0
    assert not win._records
    win.close()  # must hide, not destroy


def test_region_selector_geometry(qapp):
    from PySide6.QtCore import QRect, Qt

    from maiocr.ui.selector import RegionSelector

    sel = RegionSelector()
    sel.setGeometry(QRect(0, 0, 800, 600))
    assert int(sel.windowFlags()) & int(Qt.WindowStaysOnTopHint)
    # Regression guard: Qt.Tool windows are auto-hidden by the Windows
    # platform layer on app deactivation, which made the overlay vanish
    # mid-selection. The window type must be a plain top-level Window.
    assert sel.windowType() == Qt.Window


def test_tray_shortcut_labels_update(qapp):
    from maiocr.ui.tray import TrayIcon

    tray = TrayIcon()
    tray.update_shortcut_labels("Win+Shift+PrtSc", "Win+Ctrl+PrtSc")

    texts = [a.text() for a in tray.contextMenu().actions()]
    assert any("Win+Shift+PrtSc" in t and "文字" in t for t in texts)
    assert any("Win+Ctrl+PrtSc" in t and "代码" in t for t in texts)
    assert "Win+Shift+PrtSc" in tray.toolTip()


def test_tray_menu_actions(qapp):
    from maiocr.ui.tray import TrayIcon

    calls = []
    tray = TrayIcon(
        on_text=lambda: calls.append("text"),
        on_code=lambda: calls.append("code"),
        on_show_history=lambda: calls.append("history"),
        on_quit=lambda: calls.append("quit"),
    )
    texts = [a.text() for a in tray.contextMenu().actions()]
    assert any("Win+PrtSc" in t and "文字" in t for t in texts)
    assert any("Win+Ctrl+PrtSc" in t and "代码" in t for t in texts)
    assert any("历史记录" in t for t in texts)
    # No separate "region" entry anymore: both screenshot actions ARE the
    # unified region-selection flow.
    assert not any("区域" in t or "屏幕识别" in t for t in texts)


def test_tray_menu_has_autostart_toggle_and_restart(qapp):
    from maiocr.ui.tray import TrayIcon

    tray = TrayIcon(autostart_initial=False, on_autostart_toggled=lambda v: None)
    assert tray.autostart_action.isCheckable()
    assert not tray.autostart_action.isChecked()

    texts = [a.text() for a in tray.contextMenu().actions()]
    assert any("开机自启" in t for t in texts)
    assert any("重启" in t for t in texts)

    # Restart sits before Quit; autostart between History and Restart.
    order = [t for t in texts if t]
    assert order.index(next(t for t in order if "重启" in t)) < order.index(
        next(t for t in order if "退出" in t)
    )
    assert order.index(next(t for t in order if "历史记录" in t)) < order.index(
        next(t for t in order if "开机自启" in t)
    )


def test_tray_autostart_initial_state_and_callback(qapp):
    from maiocr.ui.tray import TrayIcon

    seen = []
    tray = TrayIcon(on_autostart_toggled=seen.append, autostart_initial=True)
    assert tray.autostart_action.isChecked()

    tray.autostart_action.setChecked(False)
    assert seen == [False]

    tray.autostart_action.setChecked(True)
    assert seen == [False, True]


def test_tray_autostart_failure_reverts(qapp):
    from maiocr.ui.tray import TrayIcon

    def denied(_enabled):
        raise OSError("access denied")

    tray = TrayIcon(on_autostart_toggled=denied, autostart_initial=False)
    tray.autostart_action.setChecked(True)
    assert not tray.autostart_action.isChecked()


def test_tray_restart_action(qapp):
    from maiocr.ui.tray import TrayIcon

    calls = []
    tray = TrayIcon(on_restart=lambda: calls.append("restart"))
    tray.restart_action.trigger()
    assert calls == ["restart"]


def _wait_until(qapp, predicate, timeout=5.0):
    """Pump Qt events while waiting (queued signals need event processing)."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


def test_triggers_route_through_region_selection(qapp, monkeypatch):
    """Hotkey/tray triggers must open the selector, then OCR that region."""
    from threading import Thread

    from PySide6.QtCore import QRect

    from maiocr.app import OCRController
    from maiocr.core.pipeline import Mode
    from maiocr.ui import selector as selector_mod

    controller = OCRController()

    runs = []

    class FakePipeline:
        def run(self, mode=Mode.TEXT, image=None, region=None):
            from maiocr.core.pipeline import PipelineResult

            runs.append((mode, region))
            return PipelineResult(mode=mode, raw_text="x", output="out")

    controller._pipeline = FakePipeline()

    rect = QRect(10, 20, 300, 200)
    monkeypatch.setattr(
        selector_mod.RegionSelector,
        "select_region",
        staticmethod(lambda: rect),
    )

    # Simulate the hotkey thread calling into the controller: the signal is
    # queued to the GUI thread, where the selector opens and OCR spawns.
    Thread(target=controller.trigger_text, daemon=True).start()
    assert _wait_until(qapp, lambda: bool(runs)), "pipeline never ran"
    assert runs[-1] == (Mode.TEXT, rect)

    # A cancelled selection must not spawn any work.
    before = len(runs)
    monkeypatch.setattr(
        selector_mod.RegionSelector,
        "select_region",
        staticmethod(lambda: None),
    )
    Thread(target=controller.trigger_code, daemon=True).start()
    assert _wait_until(qapp, lambda: len(runs) == before + 0, timeout=0.5)
    import time

    time.sleep(0.2)
    qapp.processEvents()
    assert len(runs) == before


def test_trigger_while_selecting_is_ignored(qapp, monkeypatch):
    from maiocr.app import OCRController
    from maiocr.ui import selector as selector_mod

    controller = OCRController()
    controller._pipeline = object()  # mark as ready

    entered = []
    monkeypatch.setattr(
        selector_mod.RegionSelector,
        "select_region",
        staticmethod(lambda: entered.append(1)),
    )

    # While a selection overlay is open, further requests are dropped.
    controller._selecting = True
    controller.trigger_text()          # direct delivery (GUI thread)
    qapp.processEvents()
    assert not entered

    # Once the overlay closes, the next request goes through.
    controller._selecting = False
    controller.trigger_text()
    qapp.processEvents()
    assert entered


def test_controller_busy_and_signals(qapp, monkeypatch):
    import time

    from PySide6.QtCore import QRect

    from maiocr.app import OCRController
    from maiocr.core.pipeline import Mode
    from maiocr.ui import selector as selector_mod

    controller = OCRController()
    controller.start()
    deadline = time.time() + 30
    while not controller.ready and time.time() < deadline:
        time.sleep(0.2)
    assert controller.ready, "pipeline failed to initialize"

    received = []
    controller.result_ready.connect(lambda r: received.append(r))

    class FakePipeline:
        def run(self, mode=Mode.TEXT, image=None, region=None):
            from maiocr.core.pipeline import PipelineResult

            return PipelineResult(
                mode=mode, raw_text="x", output="fake output"
            )

    monkeypatch.setattr(controller, "_pipeline", FakePipeline())
    monkeypatch.setattr(
        selector_mod.RegionSelector,
        "select_region",
        staticmethod(lambda: QRect(0, 0, 100, 50)),
    )
    controller.trigger_text()

    assert _wait_until(qapp, lambda: bool(received), timeout=5)
    assert received[0].output == "fake output"
