from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QTreeWidgetItem, QCheckBox, QMenu
from PyQt6.QtGui import QAction,QIcon
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import qtawesome as qta

from OES_toolbox.file_handling import FileLoader, SpectraDataset
from OES_toolbox.logger import Logger

from typing import TYPE_CHECKING
from collections.abc import Callable
if TYPE_CHECKING:
    from numpy.typing import ArrayLike

class SpectrumTreeItem(QTreeWidgetItem):
    """A QTreeWidgetItem representing a single spectrum"""

    ignored_files = [".png", ".jpg", ".ico", ".svg", ".pdf", ".ipynb", ".py", ".pyc"]
    ignored_prefix = ["_","."]
    logger = Logger(instance=None, context={"class":"SpectrumTreeItem"})

    _ICON_FOLDER = qta.icon("mdi6.folder")
    _ICON_FILE = qta.icon("mdi6.file-outline",color="gray")
    _ICON_FILE_CACHED = qta.icon("mdi6.file-outline","mdi6.check-bold", color="black")
    _ICON_BG = qta.icon("mdi.layers")
    _ICON_IO_ERROR = qta.icon("mdi6.file-outline","ei.remove", options=[{"color":"gray"},{"color":"red"}])
    _ICON_BG_ACTIVE = qta.icon("mdi6.file-minus-outline")

    def __init__(self,path: Path, label: str, content_num: int|None=None, is_content: bool = True,**kwargs):
        
        super().__init__(None)
        self.path = Path(path)
        self.label = label
        self.content_num = content_num
        self.is_content = is_content

        self._internal_bg = 0 # placeholder for actual internal backgrounds
        self._external_bg = 0
        self._x = kwargs.pop("x",None)
        self._y = kwargs.pop("y",None)
        # self.bg = kwargs.pop('bg',0 if self._y is not None else None)

        self.graph = pg.PlotDataItem(x=np.zeros(1), y=np.zeros(1), name=self.label, skipFiniteCheck=True)
        self._data_has_been_loaded = False        
        self.shift = 0
    
        self.setText(0,self.name())
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(0, Qt.CheckState.Unchecked)
        self._cb_shift = None # handle for shift callback/connection
        if self.path.is_dir():
            self.setIcon(0,self._ICON_FOLDER)
        if self.path.is_file() & (not self.is_content): # Not parented yet, so cannot check `self.is_file_node_item`
            self.setIcon(0,self._ICON_FILE)

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
    
    def is_active(self,with_descendants=False,with_ancestors=False):
        """Determines if an item is active, by checking it and it's decendants/or ancestors.
        
        Takes into account the current plot settings in the user interface.
        """
        mw = self.treeWidget().window()
        plot_selected = mw.plot_combobox.currentIndex()==0
        active = self.isSelected() if plot_selected else self.checked
        # Different options OR'd to check in both directions along the tree
        if with_descendants and plot_selected:
            active = active | self._is_selected_with_descendants()
        if with_descendants and not plot_selected:
            active = active | self._is_checked_with_descendants()
        if with_ancestors and plot_selected:
            active = active | self._is_selected_with_ancestors()
        if with_ancestors and not plot_selected:
            active = active | self._is_checked_with_ancestors()
        return active
    
    def name(self, shorten=False):
        if self.is_dir:
            name_stem = f"/{self.path.name}"
        elif self.is_content:
            if self.is_file_node_item and (self.label=="") or self.label == self.path.name:
                name_stem = f"/{self.path.name}"
            else:
                name_stem = f": {self.label} {self.content_num if self.content_num is not None else ''}"
        elif (self.is_file & self.is_file_node_item):
            name_stem = f"/{self.path.name}"
        else:
            name_stem = f"{self.label}"

        if shorten:
            return name_stem.strip().strip("/").lstrip(":").strip()
        parent_name = f"{self.parent().name()}" if self.parent() is not None else ""
        full_name =  f"{parent_name}{name_stem}".strip().strip("/").strip(":")
        return full_name
    
    def is_plotted(self, plot):
        return self.graph in plot.allChildItems()
    
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
    def calib(self)->Callable|None:
        """Property that looks up the current calibration on the main window, which may be None
        
        If a calibration is loaded it will be a scipy BSpline, which is a Callable, with signature `y=f(x)`.
        
        If the checkbox in the main window is checked, the calibration will be used, else, returns None.
        """
        mw = self.treeWidget().window()
        calib = getattr(mw,"cal",None)
        if mw.apply_cal_check.checkState()==Qt.CheckState.Checked:
            return calib

    @property
    def x(self):
        return self._x + self.shift

    @property
    def y(self):
        y = self._y-self.bg
        return np.nan_to_num(y /self.calib(self.x),posinf=0,neginf=0) if self.calib is not None else y

    @property
    def bg(self):
        """The background of a spectrum, composed by an internal and external background

            * Internal background: from the same file, explicitly set as the background signal
            * External background: some other SpectrumTreeItem (optional)

        Notes: 
        
        When using the external background, avoid subtracting self._external_bg.spectrum[1] as this will zero out when applying background.

        If the external background does not broadcast to the shape of the data (or is not a scalar) it will be ignored.

        It is implicitly assumed that the internal background is of correct dimensions, as it is tightly associated with the data.
        """
        #TODO: revisit this mechanism of internal+external background
        if self._external_bg==self: # Avoid recursion when checking with self            
            # ignore interal bg, else y-bg!=0
            return self._y # -self._internal_bg
        bg_ext = self._external_bg._y-self._external_bg._internal_bg if isinstance(self._external_bg,SpectrumTreeItem) else self._external_bg
            
        if (np.shape(bg_ext) == np.shape(self._y)) or (np.shape(bg_ext)==0):
            return self._internal_bg+bg_ext
        return self._internal_bg

    @property
    def spectrum(self):
        return self.graph.getData()
    
    @property
    def is_loaded(self):
        return self._data_has_been_loaded
    
    @is_loaded.setter
    def is_loaded(self, state:bool):
        self._data_has_been_loaded = state
        if state & self.is_file_node_item:
            self.setIcon(0,self._ICON_FILE_CACHED)

    @property
    def is_file_node_item(self):
        """Property to flag if a node in the tree is at the 'file-level', rather than more deeply nested.
        
        The property `self.is_file` will be `True` for any node with data (for multidimensional data).
        As a consequence, it does not work to distinguish between nodes that point to files, and nodes that point to data in a file.

        However, for any top level file, the parent (if present) will be a dir, or (if no parent present), it *would be* a dir.

        Note that for this to work reliably, self must be added to either a parent or a TreeItemWidget.
        """
        if self.is_dir:
            return False
        is_parent_a_dir = self.parent().is_dir if self.parent() is not None else True
        return (not self.is_content) or is_parent_a_dir

    def set_spectrum(self, x, y, bg=0, **kwargs):
        """Set the spectrum for this item by providing x,y and background.
        
        This method is mainly intended to set the original spectrum with data from a file, before further processing.

        The provided data will be plotted if `self.graph` is associated with a PlotWidget (the graph data will be updated regardless).
        """
        name = kwargs.pop("name", None)
        self.shift = kwargs.pop("shift", 0)
        if name:
            self.label = name
        self._x = x#[~np.isnan(x)]
        self._y = y#[~np.isnan(y)]
        self._internal_bg = bg
        self.graph.setData(x, y, skipFiniteCheck=True, name=f"file: {self.name()}", **kwargs)
        self.is_loaded = True

    def set_background(self, bg):
        """"Update the (internal, or external) background, or clear it, for the spectrum and update the plot.

        Will traverse the hierarchy down to descendants, if present.

        Will update the icon of the previous external background (if present), based on if it has a background itself, or is a file.

        Will mark the current background item (if an external SpectrumTreeItem) with an icon and statustip.

        If a file is marked for a background, which is not loaded, this will be loaded first.
        """
        # Assume external background when another SpectrumTreeItem is provided.
        is_external = isinstance(bg,SpectrumTreeItem)
        if is_external and not bg.is_loaded:
            bg.load_data()
            self.logger.info("Had to load external background file first. This could lead to unexpected backgrounds depending in file content.")
            if bg.childCount()>0:
                while bg.childCount()>0:
                    bg = bg.child(0)
                self.logger.warning("Attempting to select a background from not-yet-loaded file that contained multiple spectra. In the future, first load the file to guarantee the correct spectrum is used.")
        # Assume external background is to be cleared when None
        clear_external_bg = bg is None
        if clear_external_bg:
            self.setIcon(0,self._ICON_FILE_CACHED if self.is_file_node_item else QIcon())
        if self.childCount()>0:
            if is_external or clear_external_bg:
                if clear_external_bg and isinstance(self._external_bg,SpectrumTreeItem):
                    self._external_bg.setIcon(0,self._ICON_BG_ACTIVE if isinstance(self._external_bg._external_bg,SpectrumTreeItem) else QIcon()) 
                    self._external_bg.setStatusTip(0,None)
                # The line below will cause improperly shaped backgrounds to still be marked as background on a file node without children
                # Since no data is stored on a file node with children, we don't know ahead of time if shapes match
                # Plus the shape of each child may be different, so in some cases they may match
                # The childen will mark this correctly (if the parent is not selected)
                # Conclusion: this behaviour seems like a minor issue that would be hard to avoid in a general sense.
                self._external_bg = bg 
            for i in range(self.childCount()):
                self.child(i).set_background(bg)
            return
        # From here leaf nodes without children
        bg_values = 0 if clear_external_bg else (bg.y - bg._internal_bg if is_external else bg)
        if not (np.shape(bg_values)==np.shape(self._y) or len(np.shape(bg_values))==0):
            self.logger.info(f"Cannot set background, inappropriate shape: {np.shape(bg_values)=} vs. {np.shape(self._y)=}")
            return
        # only updating backgrounds when shape matches, or is a constant.
        if is_external or clear_external_bg:
            if self == self._external_bg and clear_external_bg:
                self.setIcon(0,self._ICON_FILE_CACHED if self.is_file_node_item else QIcon())
                self.setStatusTip(0,None)
            elif isinstance(self._external_bg,SpectrumTreeItem) and self.is_active():
                icon = QIcon()
                if isinstance(self._external_bg._external_bg,SpectrumTreeItem):
                    icon = self._ICON_BG_ACTIVE
                elif self._external_bg.is_file_node_item:
                    icon = self._ICON_FILE_CACHED                        
                self._external_bg.setIcon(0,icon) 
                self._external_bg.setStatusTip(0,None)
            self._external_bg = bg # finally update background
            if is_external:
                bg.setIcon(0,self._ICON_BG)
                self.setIcon(0,self._ICON_BG_ACTIVE)
                bg.setStatusTip(0,"Active background spectrum")
        else:
            self._internal_bg = bg_values
        self.graph.setData(self.x, self.y)


    def add_to_graph(self, plot=None):
        """Adds a graph to the central spectrum plot widget, unless another plot is provided."""
        if self.childCount() == 0:
            plot = self.treeWidget().window().specplot if plot is None else plot
            if (self.graph not in plot.allChildItems()) & (self.is_content):
                plot.addItem(self.graph)
                self.shift_wavelength(plot.window().wl_shift.value())
                # window lookup must use the plot or self.treeWidget()
                self._cb_shift = plot.window().wl_shift.sigValueChanged.connect(self.shift_wavelength)
        else:
            for i in range(self.childCount()):
                self.child(i).add_to_graph(plot)

    def remove_from_graph(self, plot=None):
        """Removes a line graph from its corresponding plot widget.
        
        Passing another `plot` as an argument only makes sense if it is part of this plot.
        """
        plot = self.graph.getViewWidget() if plot is None else plot
        if self.childCount()==0:
            if plot is not None:
                plot.removeItem(self.graph)
                if self._cb_shift is not None:
                    plot.window().wl_shift.disconnect(self._cb_shift)
        else:
            for i in range(self.childCount()-1,-1,-1):
                self.child(i).remove_from_graph()

    def remove(self, *args):       
        if self.parent() is not None:
            self.setSelected(False)
            self.remove_from_graph()
            self.parent().removeChild(self)
            self.logger.debug(f"Removing {self.name()}; children: {self.childCount()}")
        else:
            tree = self.treeWidget()
            tree.takeTopLevelItem(tree.indexOfTopLevelItem(self))
            self.clear_children()
            self.logger.debug(f"Removing top level node {self}->{self.label}")

    def shift_wavelength(self, sender):
        if self.childCount()==0:
            new = sender.value() if not isinstance(sender,float) else sender
            self.shift = new
            if self.x is not None:
                self.graph.setData(self.x, self.y, skipFiniteCheck=True)
                

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
            for f in sorted(self.path.iterdir(), key=lambda p: (p.is_file(),p.stem.lower())):
                if (f.suffix in self.ignored_files) | (f.name[0] in self.ignored_prefix):
                    continue
                subitem = SpectrumTreeItem(path=f, label=f.name, is_content=self.is_content, )
                self.addChild(subitem)
                subitem.iterdir()

    def load_data(self):
        """Load data and add appropriate amount of children, if it is not too deeply nested."""
        
        if self.is_file:
            try:
                datasets = FileLoader.open_any_spectrum(self.path.resolve())
            except (AttributeError,UnboundLocalError,EncodingWarning) as e:
                self.setIcon(0,self._ICON_IO_ERROR)
                self.logger.exception("Could not open file: %s",self.path.name)
                # raise e
            if len(datasets)>1:
                # Figure out which children already exists and update their data, rather then remove-then-add
                children = {self.child(i).label:self.child(i) for i in range(self.childCount())}
                for _i,dataset in enumerate(datasets):
                    child = children.get(dataset.name)
                    if child is None:
                        child = SpectrumTreeItem(path=self.path,is_content=True, label=dataset.name)
                        self.addChild(child)
                    child._populate_with_data(dataset, label="spectrum")
            else:
                self._populate_with_data(datasets[0], label="spectrum")
            self.is_loaded = True

    def _populate_with_data(self, dataset:SpectraDataset, label=None):
        """Add data from a SpectraDataset to this object.

        If the SpectraDataset contains multiple spectra, add necessary child items to represent the data.
        
        Accounts for current wavelength shift set in the UI when adding spectra, but it must have been added to the TreeItemWidget container in advance, because `self.treeWidget()` otherwise returns `None`.
        """
        x = dataset.x
        y = dataset.y
        tree = self.treeWidget()
        shift = tree.window().wl_shift.value() if tree is not None else 0
        
        if np.ndim(y)> 1:
            children = {self.child(i).content_num:self.child(i) for i in range(self.childCount())}
            for i in range(y.shape[1]):
                child = children.get(i)
                if child is None:
                    child = SpectrumTreeItem(self.path,label=dataset.name if label is None else label,content_num=i, is_content=True)
                    self.addChild(child)
                child.set_spectrum(
                    x,
                    y[:,i],
                    shift=shift,
                    bg = dataset.background
                )
                child.set_background(None) # clear exisiting external backgrounds if any, relevant for pre-existing children.
            self.is_loaded = True
        else:
            self.is_content = True # needed to force non-nested files to plot their data, without adding a child node.
            self.set_spectrum(x,y,shift=shift,bg=dataset.background,name=None,bg_is_internal=dataset.has_background)
        self.set_background(None)
