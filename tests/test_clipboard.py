"""Clipboard backend tests."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_empty_text_rejected(qapp):
    from maiocr.utils.clipboard import copy_text

    assert not copy_text("")
    assert not copy_text("   \n")


def test_qt_backend_copies(qapp):
    from PySide6.QtGui import QGuiApplication

    from maiocr.utils.clipboard import _copy_qt

    assert _copy_qt("maiocr-test-内容")
    assert QGuiApplication.clipboard().text() == "maiocr-test-内容"


def test_copy_text_returns_true(qapp):
    from maiocr.utils.clipboard import copy_text

    assert copy_text("hello 你好")
