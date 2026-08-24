from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from maiocr.utils.logger import get_logger

logger = get_logger()


def _icon() -> QIcon:
    """Bundled icon when available, otherwise a simple painted fallback."""
    from maiocr.utils.paths import icon_path

    path = icon_path()
    if path is not None:
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
        logger.warning("Icon file failed to load: {}", path)

    logger.info("Using fallback tray icon")
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap

    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#2d7dd2"))
    painter.setPen(QColor("#1b5e9e"))
    painter.drawRoundedRect(QRect(2, 2, 60, 60), 12, 12)

    font = painter.font()
    font.setPixelSize(30)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(pixmap.rect(), int(Qt.AlignCenter), "M")
    painter.end()

    return QIcon(pixmap)


class TrayIcon(QSystemTrayIcon):
    """
    System tray entry point.

    Both screenshot actions (and the global hotkeys) go through the same
    drag-to-select region overlay; the mode only decides how the captured
    region is post-processed (plain text vs Markdown code block).
    """

    def __init__(
        self,
        on_text=None,
        on_code=None,
        on_show_history=None,
        on_restart=None,
        on_quit=None,
        autostart_initial: bool | None = None,
        on_autostart_toggled=None,
    ):
        super().__init__(_icon())
        self.setToolTip(
            "MaiOCR\n"
            "Win+PrtSc 框选识别文字\n"
            "Win+Ctrl+PrtSc 框选识别代码\n"
            "双击查看历史记录"
        )

        self._autostart_callback = on_autostart_toggled
        self._actions: list[QAction] = []
        menu = QMenu()

        self.text_action = self._add_action(menu, "截图识别文字 (Win+PrtSc)", on_text)
        self.code_action = self._add_action(
            menu, "截图识别代码 (Win+Ctrl+PrtSc)", on_code
        )
        menu.addSeparator()
        self.history_action = self._add_action(menu, "历史记录", on_show_history)
        menu.addSeparator()
        self.autostart_action = QAction("开机自启动", menu)
        self.autostart_action.setCheckable(True)
        if autostart_initial is not None:
            self.autostart_action.setChecked(autostart_initial)
        if on_autostart_toggled is not None:
            self.autostart_action.toggled.connect(self._on_autostart_toggled)
        menu.addAction(self.autostart_action)
        self._actions.append(self.autostart_action)
        self.restart_action = self._add_action(menu, "重启 MaiOCR", on_restart)
        menu.addSeparator()
        self._add_action(menu, "退出", on_quit)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def _on_autostart_toggled(self, checked: bool):
        if self._autostart_callback is None:
            return
        try:
            self._autostart_callback(bool(checked))
        except Exception as e:
            logger.exception("Setting autostart failed: {}", e)
            self.autostart_action.blockSignals(True)
            self.autostart_action.setChecked(not checked)
            self.autostart_action.blockSignals(False)
            self.notify(f"开机自启设置失败：{e}", error=True)
        else:
            self.notify("已开启开机自启动" if checked else "已关闭开机自启动")

    def _add_action(self, menu: QMenu, text: str, callback) -> QAction:
        action = QAction(text, menu)
        if callback is not None:
            action.triggered.connect(callback)
        menu.addAction(action)
        self._actions.append(action)
        return action

    def _on_activated(self, reason):
        # Double-click opens history; single left click does nothing to
        # avoid accidental captures.
        if reason == QSystemTrayIcon.DoubleClick:
            logger.info("Tray icon double-clicked")
            self.history_action.trigger()

    def update_shortcut_labels(self, text_label: str, code_label: str):
        """Reflect the shortcuts that were actually registered."""
        self.text_action.setText(f"截图识别文字 ({text_label})")
        self.code_action.setText(f"截图识别代码 ({code_label})")
        self.setToolTip(
            "MaiOCR\n"
            f"{text_label} 框选识别文字\n"
            f"{code_label} 框选识别代码\n"
            "双击查看历史记录"
        )

    def notify(self, message: str, error: bool = False):
        self.showMessage(
            "MaiOCR",
            message,
            QSystemTrayIcon.Critical if error else QSystemTrayIcon.Information,
            2500,
        )
