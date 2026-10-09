"""Tests for persistent recognition history and restart survival."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_pipeline_result_serialization():
    from maiocr.core.pipeline import Mode, PipelineResult

    res = PipelineResult(
        mode=Mode.CODE,
        raw_text="print('hi')",
        output="```python\nprint('hi')\n```",
        language="python",
        is_code=True,
        copied=True,
        capture_ms=12.5,
        ocr_ms=45.2,
        total_ms=58.0,
        structure={"functions": 1},
        created_at="2026-10-09T10:00:00",
    )

    data = res.to_dict()
    assert data["mode"] == "code"
    assert data["raw_text"] == "print('hi')"
    assert data["output"] == "```python\nprint('hi')\n```"
    assert data["language"] == "python"
    assert data["is_code"] is True
    assert data["copied"] is True
    assert data["capture_ms"] == 12.5
    assert data["ocr_ms"] == 45.2
    assert data["total_ms"] == 58.0
    assert data["structure"] == {"functions": 1}
    assert data["created_at"] == "2026-10-09T10:00:00"

    restored = PipelineResult.from_dict(data)
    assert restored.mode == Mode.CODE
    assert restored.raw_text == res.raw_text
    assert restored.output == res.output
    assert restored.language == "python"
    assert restored.is_code is True
    assert restored.copied is True
    assert restored.capture_ms == 12.5
    assert restored.ocr_ms == 45.2
    assert restored.total_ms == 58.0
    assert restored.structure == {"functions": 1}
    assert restored.created_at == "2026-10-09T10:00:00"


def test_pipeline_result_from_dict_defaults():
    from maiocr.core.pipeline import Mode, PipelineResult

    # Missing optional fields must fall back safely
    restored = PipelineResult.from_dict({"mode": "invalid_mode", "output": "test"})
    assert restored.mode == Mode.TEXT
    assert restored.output == "test"
    assert restored.is_code is False
    assert restored.language == ""
    assert restored.structure is None
    assert bool(restored.created_at)


def test_history_persistence_across_restarts(qapp, tmp_path: Path):
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import ResultWindow

    history_file = tmp_path / "history.json"

    # --- Session 1: Add OCR results ---
    win1 = ResultWindow(history_file=history_file)
    assert win1.list_widget.count() == 0

    res1 = PipelineResult(
        mode=Mode.TEXT,
        raw_text="First entry",
        output="First entry",
        capture_ms=10.0,
        ocr_ms=20.0,
        created_at="2026-10-09T09:00:00",
    )
    res2 = PipelineResult(
        mode=Mode.CODE,
        raw_text="def foo(): pass",
        output="```python\ndef foo(): pass\n```",
        language="python",
        is_code=True,
        capture_ms=15.0,
        ocr_ms=25.0,
        created_at="2026-10-09T09:05:00",
    )

    win1.add_result(res1)
    win1.add_result(res2)

    assert win1.list_widget.count() == 2
    assert win1._records[0].output == res2.output
    assert win1._records[1].output == res1.output

    # File must exist on disk and contain both records
    assert history_file.is_file()
    saved = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert saved[0]["output"] == res2.output
    assert saved[1]["output"] == res1.output

    win1.close()
    del win1

    # --- Session 2 (Simulated App Restart): New instance loads history ---
    win2 = ResultWindow(history_file=history_file)
    assert win2.list_widget.count() == 2
    assert len(win2._records) == 2

    # Preserves order (newest first)
    assert win2._records[0].output == res2.output
    assert win2._records[0].mode == Mode.CODE
    assert win2._records[0].created_at == "2026-10-09T09:05:00"
    assert win2._records[1].output == res1.output
    assert win2._records[1].mode == Mode.TEXT
    assert win2._records[1].created_at == "2026-10-09T09:00:00"

    # Current selection is rendered in text_edit
    assert win2.text_edit.toPlainText() == res2.output

    win2.close()


def test_history_delete_selected_persists(qapp, tmp_path: Path):
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import ResultWindow

    history_file = tmp_path / "history.json"
    win = ResultWindow(history_file=history_file)

    for i in range(4):
        win.add_result(
            PipelineResult(
                mode=Mode.TEXT,
                raw_text=f"item {i}",
                output=f"item {i}",
            )
        )
    # List is [item 3, item 2, item 1, item 0]
    assert win.list_widget.count() == 4

    # Clear selection first, then select item 1 and item 2 (rows 1 and 2 in list_widget)
    win.list_widget.clearSelection()
    win.list_widget.item(1).setSelected(True)
    win.list_widget.item(2).setSelected(True)
    win.delete_selected()

    assert win.list_widget.count() == 2
    assert win._records[0].output == "item 3"
    assert win._records[1].output == "item 0"

    # Verify disk persistence
    saved = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert saved[0]["output"] == "item 3"
    assert saved[1]["output"] == "item 0"

    win.close()
    del win

    # Restart and confirm deleted items stay deleted
    win_restarted = ResultWindow(history_file=history_file)
    assert win_restarted.list_widget.count() == 2
    assert win_restarted._records[0].output == "item 3"
    assert win_restarted._records[1].output == "item 0"
    win_restarted.close()


def test_history_clear_persists(qapp, tmp_path: Path):
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import ResultWindow

    history_file = tmp_path / "history.json"
    win = ResultWindow(history_file=history_file)

    win.add_result(PipelineResult(mode=Mode.TEXT, raw_text="text", output="text"))
    assert win.list_widget.count() == 1

    win.clear_history()
    assert win.list_widget.count() == 0

    # Restart and confirm cleared state persists
    win_restarted = ResultWindow(history_file=history_file)
    assert win_restarted.list_widget.count() == 0
    assert not win_restarted._records
    win_restarted.close()


def test_history_corrupt_file_recovery(qapp, tmp_path: Path):
    from maiocr.ui.result import ResultWindow

    history_file = tmp_path / "history.json"
    # Write corrupted JSON
    history_file.write_text("{{This is corrupted json", encoding="utf-8")

    win = ResultWindow(history_file=history_file)
    # Must not crash, starts empty
    assert win.list_widget.count() == 0
    assert not win._records

    # Corrupt file must have been backed up
    backups = list(tmp_path.glob("history.json.corrupt.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{{This is corrupted json"
    win.close()


def test_history_max_limit_enforced(qapp, tmp_path: Path):
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import MAX_HISTORY, ResultWindow

    history_file = tmp_path / "history.json"
    win = ResultWindow(history_file=history_file)

    for i in range(MAX_HISTORY + 15):
        win.add_result(
            PipelineResult(mode=Mode.TEXT, raw_text=f"line {i}", output=f"line {i}")
        )

    assert win.list_widget.count() == MAX_HISTORY
    assert len(win._records) == MAX_HISTORY
    assert win._records[0].output == f"line {MAX_HISTORY + 14}"

    # Verify saved file also adheres to MAX_HISTORY
    saved = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(saved) == MAX_HISTORY
    win.close()


def test_item_label_date_formatting():
    from maiocr.core.pipeline import Mode, PipelineResult
    from maiocr.ui.result import ResultWindow

    now = datetime.now()
    today_iso = now.isoformat(timespec="seconds")
    res_today = PipelineResult(
        mode=Mode.TEXT,
        raw_text="abc",
        output="abc",
        created_at=today_iso,
    )
    label_today = ResultWindow._item_label(res_today)
    assert now.strftime("%H:%M:%S") in label_today

    yesterday = now - timedelta(days=2)
    yesterday_iso = yesterday.isoformat(timespec="seconds")
    res_old = PipelineResult(
        mode=Mode.CODE,
        raw_text="x = 1",
        output="x = 1",
        language="python",
        is_code=True,
        created_at=yesterday_iso,
    )
    label_old = ResultWindow._item_label(res_old)
    assert yesterday.strftime("%m-%d %H:%M") in label_old
    assert "代码·python" in label_old


def test_singleinstance_guard_none_handling():
    from maiocr.utils.singleinstance import SingleInstance

    guard = SingleInstance("test-guard-key")
    assert guard._server is None
    # release() when not acquired must be completely safe
    guard.release()


def test_clipboard_pyperclip_bad_fd_handling(monkeypatch):
    import os

    from maiocr.utils.clipboard import _copy_pyperclip

    def mock_dup(fd):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(os, "dup", mock_dup)
    # Should not raise OSError, handled gracefully
    result = _copy_pyperclip("test text")
    assert isinstance(result, bool)


def test_region_capture_mss_boundary_clamping(monkeypatch):
    from PySide6.QtCore import QPoint, QRect

    from maiocr.capture import screen

    grabbed = []

    def mock_grab(clipped):
        grabbed.append(clipped)
        return object()

    mock_monitor = {"left": 100, "top": 100, "width": 1920, "height": 1080}
    origin = QPoint(100, 100)
    dpr = 1.0

    monkeypatch.setattr(screen, "_match_monitor", lambda pt: (mock_monitor, origin, dpr))
    monkeypatch.setattr(screen, "_grab_mss", mock_grab)

    # 1. Normal inside region
    rect_inside = QRect(150, 150, 200, 100)
    res = screen._region_capture_mss(rect_inside)
    assert res is not None
    assert grabbed[-1]["left"] == 150
    assert grabbed[-1]["top"] == 150
    assert grabbed[-1]["width"] == 200
    assert grabbed[-1]["height"] == 100

    # 2. Region spilling to the left of monitor (left < origin.x)
    rect_left_spill = QRect(50, 150, 100, 100)  # left offset is -50, width is 100
    res = screen._region_capture_mss(rect_left_spill)
    assert res is not None
    assert grabbed[-1]["left"] == 100  # clamped to monitor left
    assert grabbed[-1]["width"] == 50  # 100 + (-50) = 50

    # 3. Region completely outside to the left
    rect_outside = QRect(0, 150, 50, 100)  # left offset is -100, width is 50 -> clamped width <= 0
    res = screen._region_capture_mss(rect_outside)
    assert res is None

