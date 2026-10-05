"""Implementation of a logger widget for the front-end, along with required components."""
import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import PyQt6.QtWidgets as QtWidgets
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal

from ._version import version
from .logger import LOGGER_NAME

level_to_name_mapping = {v:k for k,v in logging.getLevelNamesMapping().items()}


class LoggerSignals(QObject):
    log = pyqtSignal(object, str)

class FrontendLogHandler(logging.Handler):
    

    def __init__(self):
        super().__init__()
        self.signals = LoggerSignals()
        self._fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.setFormatter(self._fmt)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.signals.log.emit(record, msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)



class LogWidget(pg.LayoutWidget):
    """A widget for logging messages to the front-end."""
   
    sigLevelChange = pyqtSignal(int)

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
            logging.DEBUG: "blue",
            logging.INFO: "green",
            logging.WARNING: "orange",
            logging.ERROR: "red",
            logging.CRITICAL: "purple",
        }

    def __init__(self, parent=None):
        
        super().__init__(parent)
        self.setWindowTitle("OES Toolbox Log")
        self._logger = logging.getLogger(LOGGER_NAME)
        self.handler = next((h for h in self._logger.handlers if isinstance(h, FrontendLogHandler)), None)
        if self.handler is None:
            self.handler = FrontendLogHandler()
            self._logger.addHandler(self.handler)
            self.handler.setLevel(logging.INFO)
        self.select_level = QtWidgets.QComboBox(self)
        self.select_level.addItems(["Debug", "Info", "Warning", "Error", "Critical"])
        self.select_level.setCurrentText(logging.getLevelName(self.handler.level).title())
        self.select_level.textActivated.connect(self.on_change_level) # Always apply user choice, even if it does not look different.
        self.addLabel("Log level:")
        self.addWidget(self.select_level)
        self.btn_clear_log = QtWidgets.QPushButton("Clear log", parent=self)
        self.btn_clear_log.clicked.connect(self.on_clear_log)
        self.addWidget(self.btn_clear_log, 0, 2)
        self.btn_save_log = QtWidgets.QPushButton("Save log", parent=self)
        self.btn_save_log.clicked.connect(self.on_save_log)
        self.addWidget(self.btn_save_log,0,3)

        self.output_log = QtWidgets.QPlainTextEdit(
            "",
            parent=self)
        self.output_log.setReadOnly(True)
        self.output_log.setPlaceholderText(f"Welcome to OES toolbox {version}")
        self.output_log.setDocumentTitle("Log")
        self.output_log.setMinimumWidth(400)
        self.addWidget(self.output_log, 1, 0, 2, 4)
        self.layout.setColumnStretch(3,1)  # ty: ignore[unresolved-attribute]
        self.layout.setRowStretch(1,1)  # ty: ignore[unresolved-attribute]
        
        self.handler.signals.log.connect(self.append_log_output)

    @property
    def level(self):
        return level_to_name_mapping.get(self.handler.level,"Info")

    def append_log_output(self, record: logging.LogRecord, msg:str):
        # record, msg = log_signal_msg
        formatted_msg = f'| <span style="color:{self.LEVEL_COLORS.get(record.levelno, "black")};">{record.levelname}</span> | {msg}'
        self.output_log.appendHtml(formatted_msg)

    def on_change_level(self,level:str):
        # level = logging.getLevelName(level.upper())
        level = logging.getLevelNamesMapping().get(level.upper())
        if not level:
            return
        self.handler.setLevel(level)
        print(f"Intended level: {level}, actual: {self.level}")

    def on_clear_log(self):
        self.output_log.clear()

    def on_save_log(self):
        fname,_filter = QtWidgets.QFileDialog.getSaveFileName(self,"Save log as file",filter="Log file (*.log)")
        if fname=="":
            return
        fp = Path(fname).resolve()
        if fp.is_file():
            fp.write_text(self.output_log.toPlainText(),encoding="utf-8")

    def quit_cleanly(self):
        self.close()