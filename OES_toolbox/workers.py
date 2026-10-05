"""Module for worker classes for the OES toolbox.

For ephemeral workers that perform "once-off" or "fire-and-forget" kind of tasks (like fits), use QRunnable.

For long-running, persistent workers that require periodic callbacks etc, use a QObject-based worker moved to a QThread.

Don't override the QThread.run method, to preserve the worker event-loop.
"""
import threading
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from .calc import molecule_objective
from .lazy_import import lazy_import
from .logger import ContextLogger

lmfit = lazy_import("lmfit")

if TYPE_CHECKING:
    import pandas as pd
    from lmfit import Parameters

class WorkerSignals(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(tuple)  # (exctype, value, traceback)

class BaseWorker(QRunnable):
    """Base class for QRunnable workers with standard signals and logging."""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.logger = ContextLogger(self)
        self._abort = threading.Event()

    def abort(self):
        self._abort.set()

    def is_aborted(self):
        return self._abort.is_set()

    @pyqtSlot()
    def run(self):
        raise NotImplementedError("Subclasses must implement this method.")

class MooseFitWorker(BaseWorker):
    """Ephemeral worker for performing molecular band fits of multiple species, powered by Moose."""
    
    def __init__(self, source,x,y, params:"Parameters",molecule_dbs:"dict[str,pd.DataFrame]"):
        super().__init__()
        self.source = source
        self.x = x
        self.y = y
        self.params = params
        self.molecule_dbs = molecule_dbs

    @pyqtSlot()
    def run(self):
        self.signals.started.emit()
        try:
            # Perform the fitting operation here
            result = self._fit()
            y_fit = molecule_objective(result.params,self.x, normalize=True,**self.molecule_dbs)
            self.signals.result.emit((self.source, result, self.x, y_fit))
            
            self.signals.progress.emit(1)
        except Exception:
            self.logger.exception("An error occurred during fitting.")
        finally:
            self.signals.finished.emit()

    def _fit(self):
        result = lmfit.minimize(
            molecule_objective,
            self.params,
            args = (self.x,),
            kws = {"y": self.y, "normalize": True, **self.molecule_dbs},
            ftol = 1e-10,
            max_nfev=3000
        )
        return result
        
        





