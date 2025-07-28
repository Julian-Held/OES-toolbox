from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from PyQt6.QtWidgets import QTreeWidgetItem, QCheckBox, QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from OES_toolbox.file_handling import FileLoader
from OES_toolbox.logger import Logger


class SpectrumTreeItem(QTreeWidgetItem):
    """A QTreeWidgetItem representing a single spectrum"""

    ignored_files = [".png", ".jpg", ".ico", ".svg", ".pdf", ".ipynb", ".py", ".pyc"]
    ignored_prefix = ["_","."]
    logger = Logger(instance=None, context={"class":"SpectrumTreeItem"})

    def __init__(self,path: Path, label: str, content_num: int, is_content: bool = True):
        
        super().__init__(None)
        self.path = Path(path)
        self.label = label
        self.content_num = content_num
        self.is_content = is_content

        self.graph = pg.PlotDataItem(x=np.zeros(1), y=np.zeros(1), name=self.label, skipFiniteCheck=True)
        self._data_has_been_loaded = False
        self._x = None
        self._y = None
        self.bg = 0
        self.shift = 0
        
        if (self.path.is_dir()) or (not is_content):
            self.setText(0, self.path.relative_to(self.path.joinpath("../").resolve()).as_posix())
        else:
            self.setText(0, label)
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(0, Qt.CheckState.Unchecked)
        self._cb_shift = None

    def _is_checked_with_descendants(self):
        answer = self.checked if self.childCount()==0 else (True in [self.child(i)._is_checked_with_descendants() for i in range(self.childCount())])
        return answer | self.checked
    
    def _is_checked_with_ancestors(self):
        parent: SpectrumTreeItem|None = self.parent()
        answer = self.checked if parent is None else True is parent._is_checked_with_ancestors()
        return answer | self.checked
    
    def _is_selected_with_descendants(self):
        answer = self.isSelected() if self.childCount()==0 else (True in [self.child(i)._is_selected_with_descendants() for i in range(self.childCount())])
        return answer | self.isSelected()
    
    def _is_selected_with_ancestors(self):
        parent: SpectrumTreeItem|None = self.parent()
        answer = self.isSelected() if parent is None else (True is parent._is_selected_with_ancestors())
        return answer |  self.isSelected()
    
    
    @property
    def checked(self):
        return self.checkState(0) == Qt.CheckState.Checked

    @property
    def is_file(self):
        return self.path.is_file()
    
    @property
    def is_dir(self):
        return self.path.is_dir()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new: ArrayLike):
        self._x = new[~np.isnan(new)]
        self.graph.setData(new + self.shift, self.y - self.bg)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new: ArrayLike):
        self._y = new[~np.isnan(new)]
        self.graph.setData(self.x + self.shift, new - self.bg)

    @property
    def spectrum(self):
        return self.graph.getData()

    def set_spectrum(self, x, y, bg=None, **kwargs):
        name = kwargs.pop("name", None)
        self.shift = kwargs.pop("shift", 0)
        if name:
            self.label = name
        self._x = x[~np.isnan(x)]
        self._y = y[~np.isnan(y)]
        # self.bg = self.bg if bg is None else bg[~np.isnan(bg)]
        if bg is not None:
            self.bg = bg if len(np.shape(bg))== 0 else bg[~np.isnan(bg)]     
        self.graph.setData(x + self.shift, y - self.bg, skipFiniteCheck=True, name=f"file: {self.name()}", **kwargs)
        self._data_has_been_loaded = True

    def set_background(self, bg):
        if self.childCount()==0:
            if (np.shape(bg) == np.shape(self._y)) or (len(np.shape(bg))==0):
                self.bg = bg
                self.graph.setData(self._x + self.shift, self._y - bg)
            else:
                self.logger.info(f"Cannot set background, inapproriate shape: {np.shape(bg)=} vs. {np.shape(self._y)=}")
        else:
            for i in range(self.childCount()):
                self.child(i).set_background(bg)

    def add_to_graph(self, plot):
        if self.childCount() == 0:
            if (self.graph not in plot.allChildItems()) & (self.is_content):
                plot.addItem(self.graph)
                self.shift_wavelength(plot.window().wl_shift.value())
                self._cb_shift = plot.window().wl_shift.sigValueChanged.connect(self.shift_wavelength)
        else:
            for i in range(self.childCount()):
                self.child(i).add_to_graph(plot)

    def remove_from_graph(self, plot=None):
        plot = self.graph.getViewWidget() if plot is None else plot
        if self.childCount()==0:
            if plot is not None:
                plot.removeItem(self.graph)
                if self._cb_shift is not None:
                    plot.window().wl_shift.disconnect(self._cb_shift)
        else:
            for i in range(self.childCount()-1,-1,-1):
                self.child(i).remove_from_graph(plot)

    def remove(self, *args):       
        if self.parent() is not None:
            self.remove_from_graph()
            self.parent().removeChild(self)
            self.logger.debug(f"Removing {self}->{self.label}; children: {self.childCount()}")
        else:
            tree = self.treeWidget()
            tree.takeTopLevelItem(tree.indexOfTopLevelItem(self))
            self.clear_children()
            self.logger.debug(f"Removing top level node {self}->{self.label}")

    def shift_wavelength(self, sender):
        if self.childCount()==0:
            new = sender.value() if not isinstance(sender,float) else sender
            self.shift = new
            if self._x is not None:
                self.graph.setData(self._x + new, self._y - self.bg, skipFiniteCheck=True)

    def clear_tree(self):
        tree = self.treeWidget()
        widgets: list[SpectrumTreeItem] = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        for widget in widgets:
            widget.remove()

    def clear_children(self):
        """Remove child elements starting with the last child."""
        for i in range(self.childCount()-1,-1,-1):
            self.child(i).remove()

    def iterdir(self):
        if self.path.is_dir():
            self.logger.debug(f"Iterating over dir={self.path}")
            for f in self.path.iterdir():
                if (f.suffix in self.ignored_files) | (f.name[0] in self.ignored_prefix):
                    continue
                subitem = SpectrumTreeItem(path=f, label=f.name, content_num=0, is_content=self.is_content, )
                self.addChild(subitem)
                subitem.iterdir()

    def load_data(self):
        """Load data and add appropriate amount of children, if it is not too deeply nested."""
        
        if self.is_file:
            self.logger.info(f"Reading file {self.path}")
            datasets = FileLoader.open_any_spectrum(self.path.resolve())
            # print(f"Loaded {len(datasets)} datasets from {self.path.resolve()}")
            if len(datasets)>1:
                for i,dataset in enumerate(datasets):
                    child = SpectrumTreeItem(path=self.path,content_num=i,is_content=True, label='ROI')
                    child._populate_with_data(dataset)
                    self.addChild(child)
            else:
                self._populate_with_data(datasets[0])
            self._data_has_been_loaded = True

    def _populate_with_data(self,dataset:SpectraDataset):
        x = dataset.x
        y = dataset.y
        tree = self.treeWidget()
        shift = tree.window().wl_shift.value() if tree is not None else 0
        if np.ndim(y)> 1 and np.shape(y)[0]>1:
            for i in range(y.shape[1]):
                child = SpectrumTreeItem(self.path,label=dataset.name,content_num=i, is_content=True)
                self.addChild(child)
                child.set_spectrum(
                    x,
                    y[:,i],
                    shift=shift,
                    bg = dataset.background
                )
        elif np.ndim(y)>1:
            self.set_spectrum(x,y[0], shift=0, name=self.path.name)
        else:
            self.set_spectrum(x,y,shift=0,name=self.path.name)
