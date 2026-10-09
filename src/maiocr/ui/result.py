import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from maiocr.core.pipeline import PipelineResult
from maiocr.utils.clipboard import copy_text
from maiocr.utils.logger import get_logger
from maiocr.utils.paths import history_file_path

logger = get_logger()

MAX_HISTORY = 100


class ResultWindow(QMainWindow):
    """OCR result viewer with persistent history."""

    def __init__(self, history_file: Path | None = None):
        super().__init__()

        self.setWindowTitle("MaiOCR 识别结果")
        self.resize(720, 480)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        if history_file is not None:
            self._history_file = Path(history_file)
        elif "MAIOCR_HISTORY_FILE" in os.environ:
            self._history_file = Path(os.environ["MAIOCR_HISTORY_FILE"])
        elif "PYTEST_CURRENT_TEST" in os.environ:
            self._history_file = None
        else:
            self._history_file = history_file_path()

        self._records: list[PipelineResult] = []

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(230)
        # Enable multi-selection with standard Windows behavior
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.text_edit.setFont(font)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.text_edit)
        splitter.setStretchFactor(1, 1)

        self.status_label = QLabel("就绪")

        copy_button = QPushButton("复制")
        copy_button.clicked.connect(self.copy_current)

        delete_selected_button = QPushButton("删除选中")
        delete_selected_button.clicked.connect(self.delete_selected)
        delete_selected_button.setEnabled(False)
        self.delete_selected_button = delete_selected_button

        clear_button = QPushButton("清空历史")
        clear_button.clicked.connect(self.clear_history)

        close_button = QPushButton("关闭 (Esc)")
        close_button.clicked.connect(self.hide)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(copy_button)
        buttons.addWidget(delete_selected_button)
        buttons.addWidget(clear_button)
        buttons.addWidget(close_button)

        layout = QVBoxLayout()
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.hide)
        # Delete key to delete selected items
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)

        self._load_saved_history()

    # ------------------------------------------------------------------ API

    def add_result(self, result: PipelineResult):
        """Record a new OCR result and surface it."""
        logger.info(
            "Result added: mode={} chars={} lang={}",
            result.mode.value,
            len(result.output),
            result.language or "-",
        )

        self._records.insert(0, result)
        del self._records[MAX_HISTORY:]

        item = QListWidgetItem(self._item_label(result))
        self.list_widget.insertItem(0, item)
        while self.list_widget.count() > MAX_HISTORY:
            self.list_widget.takeItem(self.list_widget.count() - 1)

        self.list_widget.blockSignals(True)
        self.list_widget.clearSelection()
        self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)
        self._render(result)
        self._save_history()

        self.show()
        self.raise_()
        self.activateWindow()

    def show_history(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def copy_current(self):
        if not self._records:
            return
        row = max(self.list_widget.currentRow(), 0)
        index = min(row, len(self._records) - 1)
        record = self._records[index]
        if copy_text(record.output):
            self.status_label.setText(f"已复制（{len(record.output)} 字符）")

    def delete_selected(self):
        """Delete all selected history records."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        # Get indices of selected items (in descending order for safe removal)
        selected_rows = sorted(
            [self.list_widget.row(item) for item in selected_items],
            reverse=True,
        )

        for row in selected_rows:
            if 0 <= row < len(self._records):
                del self._records[row]
                self.list_widget.takeItem(row)

        # Update status
        self.list_widget.clearSelection()
        if self._records:
            # Select the item at the lowest deleted row (or last item if at end)
            new_row = min(selected_rows[-1], len(self._records) - 1)
            self.list_widget.setCurrentRow(new_row)
            self._render(self._records[new_row])
            self.status_label.setText(f"已删除 {len(selected_rows)} 条记录")
        else:
            self.text_edit.clear()
            self.status_label.setText("历史已清空")

        self._save_history()
        logger.info("Deleted {} selected history records", len(selected_rows))

    def clear_history(self):
        self._records.clear()
        self.list_widget.clear()
        self.text_edit.clear()
        self.status_label.setText("历史已清空")
        self._save_history()
        logger.info("History cleared")

    # ------------------------------------------------------------- internals

    def _load_saved_history(self):
        """Load persistent history from disk on startup."""
        if self._history_file is None or not self._history_file.is_file():
            return
        try:
            content = self._history_file.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            if not isinstance(data, list):
                logger.warning("History file is not a list: {}", self._history_file)
                return

            records: list[PipelineResult] = []
            for item in data[:MAX_HISTORY]:
                if isinstance(item, dict):
                    try:
                        records.append(PipelineResult.from_dict(item))
                    except Exception as e:
                        logger.warning("Failed to deserialize history record: {}", e)

            self._records = records
            self.list_widget.blockSignals(True)
            self.list_widget.clear()
            for rec in records:
                self.list_widget.addItem(QListWidgetItem(self._item_label(rec)))
            if records:
                self.list_widget.setCurrentRow(0)
                self._render(records[0])
            self.list_widget.blockSignals(False)
            logger.info(
                "Loaded {} history records from {}", len(records), self._history_file
            )
        except Exception as e:
            logger.warning("Failed to load history from {}: {}", self._history_file, e)
            try:
                corrupt_backup = self._history_file.with_name(
                    f"history.json.corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                self._history_file.rename(corrupt_backup)
                logger.warning("Backed up corrupted history file to {}", corrupt_backup)
            except Exception as backup_err:
                logger.warning(
                    "Failed to backup corrupted history file: {}", backup_err
                )


    def _save_history(self):
        """Persist current history records atomically to disk."""
        if self._history_file is None:
            return
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            data = [rec.to_dict() for rec in self._records[:MAX_HISTORY]]

            tmp_fd, tmp_path_str = tempfile.mkstemp(
                dir=str(self._history_file.parent),
                prefix="history_",
                suffix=".tmp",
                text=True,
            )
            tmp_path = Path(tmp_path_str)
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._history_file)
            logger.debug(
                "Saved {} history records to {}", len(data), self._history_file
            )
        except Exception as e:
            logger.error("Failed to save history to {}: {}", self._history_file, e)

    @staticmethod
    def _item_label(result: PipelineResult) -> str:
        stamp = ""
        if result.created_at:
            try:
                dt = datetime.fromisoformat(result.created_at)
                now = datetime.now()
                if dt.date() == now.date():
                    stamp = dt.strftime("%H:%M:%S")
                else:
                    stamp = dt.strftime("%m-%d %H:%M")
            except Exception:
                stamp = result.created_at[:8]
        if not stamp:
            stamp = datetime.now().strftime("%H:%M:%S")
        kind = f"代码·{result.language}" if result.is_code else "文字"
        return f"{stamp}  {kind}  {len(result.output)}字"

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._records):
            self._render(self._records[row])

    def _on_selection_changed(self):
        """Update delete button state based on selection."""
        has_selection = len(self.list_widget.selectedItems()) > 0
        self.delete_selected_button.setEnabled(has_selection)

    def _render(self, result: PipelineResult):
        self.text_edit.setPlainText(
            result.output if result.output else "（未识别到内容）"
        )
        parts = [
            result.summary,
            f"{len(result.output)} 字符",
            f"截图 {result.capture_ms:.0f}ms",
            f"OCR {result.ocr_ms:.0f}ms",
        ]
        if result.output.strip():
            parts.append("已复制到剪贴板" if result.copied else "复制失败")
        self.status_label.setText(" · ".join(parts))

    def closeEvent(self, event):
        event.ignore()
        self.hide()

