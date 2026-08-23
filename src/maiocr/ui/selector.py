from PySide6.QtCore import QEventLoop, QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from maiocr.utils.logger import get_logger

logger = get_logger()

MIN_SELECTION_SIZE = 6


class RegionSelector(QWidget):
    """
    Fullscreen translucent overlay for dragging out a capture rectangle.

    Use `select_region()` which blocks until the user finishes or cancels.
    Returns a global-logical QRect or None when cancelled.
    """

    def __init__(self):
        super().__init__(None)

        self._origin = None
        self._current = None
        self._loop: QEventLoop | None = None
        self._result: QRect | None = None
        self._closing = False

        # Qt.Window, NOT Qt.Tool: the Windows platform plugin hides tool
        # windows automatically whenever the application loses activation,
        # which made this overlay flash up and instantly disappear if
        # anything stole focus right after the hotkey fired.
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Window
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ API

    @staticmethod
    def select_region() -> QRect | None:
        app = QGuiApplication.instance()
        if app is None:
            logger.error("Region selector requires QApplication")
            return None

        geo = QRect()
        for screen in app.screens():
            geo = geo.united(screen.geometry())
        if geo.isEmpty():
            return None

        selector = RegionSelector()
        selector.setGeometry(geo)
        selector._result = None
        selector.show()
        selector.raise_()
        selector.activateWindow()
        selector.setFocus()

        loop = QEventLoop()
        selector._loop = loop
        loop.exec()
        return selector._result

    # ------------------------------------------------------------- painting

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        # Usage hint while nothing is being dragged yet.
        if self._origin is None:
            painter.setPen(QColor(255, 255, 255, 200))
            hint = "拖拽框选识别区域 · Esc 或右键取消"
            painter.drawText(
                self.rect().adjusted(0, 24, 0, 0),
                Qt.AlignHCenter | Qt.AlignTop,
                hint,
            )

        band = self._band_rect()
        if band is not None:
            painter.setCompositionMode(
                QPainter.CompositionMode_Clear
            )
            painter.drawRect(band.adjusted(-1, -1, 1, 1))

            painter.setCompositionMode(
                QPainter.CompositionMode_SourceOver
            )
            painter.setPen(QColor(255, 255, 255, 220))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(band)

            size_label = f"{band.width()} x {band.height()}"
            painter.setPen(QColor(255, 255, 255, 230))
            painter.drawText(
                band.adjusted(0, -22, 0, -4),
                Qt.AlignLeft,
                size_label,
            )

    def _band_rect(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    # ---------------------------------------------------------------- events

    def hideEvent(self, event):
        super().hideEvent(event)
        # Safety net: if the overlay is hidden by anything outside our
        # control, cancel cleanly instead of leaving the caller blocked in
        # the nested event loop forever.
        if not self._closing:
            logger.info("Region selection aborted (overlay hidden externally)")
            self._finish(None)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            logger.info("Region selection cancelled (Esc)")
            self._finish(None)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()
        elif event.button() == Qt.RightButton:
            self._finish(None)

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._origin is None:
            return

        self._current = event.position().toPoint()
        band = self._band_rect()

        if band is None or band.width() < MIN_SELECTION_SIZE or band.height() < MIN_SELECTION_SIZE:
            logger.info("Region selection cancelled (too small)")
            self._finish(None)
            return

        band.translate(self.geometry().topLeft())
        logger.info("Region selected: {}", band.getRect())
        self._finish(band)

    # --------------------------------------------------------------- helpers

    def _finish(self, result: QRect | None):
        self._closing = True
        self._result = result
        self.hide()
        self.deleteLater()

        if self._loop is not None:
            self._loop.quit()
            self._loop = None
