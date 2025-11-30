import os
import sys
from pathlib import Path
import platform
import subprocess
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QFileDialog, QTreeWidgetItem, \
        QTreeWidgetItemIterator , QHeaderView, \
        QMainWindow, QVBoxLayout, QToolButton, QDialog, \
        QDialogButtonBox, QLabel, QMenu,QTreeWidget,QInputDialog, \
        QProgressBar,QMessageBox
from PyQt6.QtCore import Qt, QSettings, \
        QStandardPaths, QFile
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6 import sip, QtGui
import pyqtgraph as pg
import webbrowser
import qtawesome as qta

file_dir = os.path.dirname(os.path.abspath(__file__))

from .ui import resources # seems unused but is needed!
from OES_toolbox.settings import settings
from OES_toolbox.fio import fio
from OES_toolbox.ident import ident_module
from OES_toolbox.molecules import molecule_module
from OES_toolbox.continuum import cont_module
from OES_toolbox.Widgets import SpectrumTreeItem
from OES_toolbox.logger import Logger
from OES_toolbox.lazy_import import lazy_import
from OES_toolbox.file_handling import FileLoader

from importlib.metadata import metadata
scipy = lazy_import("scipy")

colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # matplotlib default
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True)


INVALID_CALIB_TXT = (
    "Invalid calibration file format",
    ("Invalid or unreadable calibration file.\n"
    "Please make sure that the file is a text file which contains only two columns.\n"
    "The wavelength must be in nm and the sensitivity of the spectrometer in arbitrary units, representing the photon count per second.\n"
    "The file must use a point as the decimal character and either a tab or comma as the delimiter between the two columns."
))


class about_dialog(QDialog):
    def __init__(self):
        super().__init__()
        m = metadata("OES_toolbox")
        self.setWindowTitle("About OES toolbox")

        msg = f"""OES toolbox - Helping out with optical emission spectroscopy of low-temperature plasmas.
        Powered by owl, Moose/MassiveOES, astroquery and others.
        Version: {m['version']}
        {m['License-Expression']} License - Copyright (c) 2024 Julian Held
        """
        for url in m.get_all("Project-URL"):
            category,link = url.split(', ')
            msg += f"{category}: {link}\n"

        QBtn = QDialogButtonBox.StandardButton.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel(msg)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(file_dir + "/ui/main.ui", self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,0)
        self.progress_bar.setMaximumWidth(180)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.status_msg = QLabel()
        self.statusBar().addPermanentWidget(self.status_msg)
        self.logger = Logger(self)
        
        self.settings = settings(self)
        self.io = fio(self)
        self.mol = molecule_module(self)
        self.ident = ident_module(self)
        self.cont = cont_module(self)
        
        # prepare config and data files
        self.conf = QSettings("OES toolbox", "OES toolbox")
        self.roaming_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        self.cal_path = os.path.join(self.roaming_path, 'calibration')
        if not os.path.exists(self.roaming_path):
            os.makedirs(self.roaming_path)
        if not os.path.exists(self.cal_path):
            os.makedirs(self.cal_path)
        self.cal = None
        QTimer.singleShot(200, self.cal_files_refresh) # TODO move out of thread to improve startup perfromance
        self.max_child_plot = 8
        
        # center plot
        self.specplot.setLabel("left", "intensity")
        self.specplot.setLabel("bottom", "wavelength / nm")
        self.specplot.setAxisItems({"top": pg.AxisItem("top",linkView=self.specplot.getViewBox()),"right":pg.AxisItem("right",linkView=self.specplot.getViewBox(),)})
        self.specplot.addLegend()
        self.copy_plots_btn.clicked.connect(self.action_graph_to_clipboard.trigger)
        self.action_graph_to_clipboard.triggered.connect(self.graph_to_clipboard)
        self.action_export_plot_data.triggered.connect(self.io.save_plots)
        self.actionRefresh_plots.triggered.connect(self.update_spec)
        self.proxy = pg.SignalProxy(self.specplot.scene().sigMouseMoved, rateLimit=90, slot=self.update_plot_pos)
        self.actionClear_Plots.triggered.connect(self.clear_all_spec)
    
        # file loading, plotting /drag & drop
        self.button_open.clicked.connect(self.actionOpenFolder.trigger)
        self.button_open_files.clicked.connect(self.actionOpenFiles.trigger)
        self.actionOpenFolder.triggered.connect(self.io.open_folder)
        self.actionOpenFiles.triggered.connect(self.io.open_files)
        
        self.file_list.itemSelectionChanged.connect(self.on_selection_change)
        self.file_list.itemChanged.connect(self.on_check_change)
        self.plot_combobox.currentIndexChanged.connect(self.update_spec)
        self.file_list.dropEvent = self.do_drag_drop
        self.file_list.dragEnterEvent = self.check_drag_drop
        self.file_list.dragMoveEvent = self.check_drag_drop
        self.file_list.currentItemChanged.connect(self.on_current_item_changed)
        self.file_list.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.clear_file_list.clicked.connect(self.actionClearFiles.trigger)
        self.actionClearFiles.triggered.connect(self.on_file_clear_action)
        # self.file_list.keyPressEvent = self.io.file_list_keys
        self.file_list.keyPressEvent = self.file_list_keys
        self.bg_internal_check.hide()
        self.bg_internal_check.stateChanged.connect(self.update_spec)
        # self.bg_extra_btn.clicked.connect(self.io.open_bg_file)
        # self.bg_extra_btn.clicked.connect(self.on_open_bg_file)
        self.bg_extra_check.stateChanged.connect(self.on_bg_check_change)
        self.reload_file_btn.clicked.connect(lambda: self.on_reload_file_action(self.file_list.selectedItems()[0]))
        self.clear_file_btn.clicked.connect(self.on_file_clear_action)
        
        # self.bg_extra_ledit.num = 0
        self.file_list.customContextMenuRequested.connect(self.file_rightClick)
        self.file_list.viewport().installEventFilter(self)

        # help menu
        self.actionDocumentation.triggered.connect(lambda: webbrowser.open('https://github.com/mimurrayy/OES-toolbox/wiki'))
        self.actionHow_to_cite.triggered.connect(lambda: webbrowser.open('https://github.com/mimurrayy/OES-toolbox/wiki/How-to-cite'))
        self.actionAbout.triggered.connect(lambda: about_dialog().exec())

        # spectromter settings
        self.groupBox_6.hide() # TODO not a thing yet, hide for now
        self.specopt_axis_group.hide()
        # intensity calibration    
        self.open_cal_folder_btn.clicked.connect(self.open_cal_folder)
        self.add_cal_file_btn.clicked.connect(self.add_cal_file)
        self.cal_refresh_btn.clicked.connect(self.cal_files_refresh)
        self.apply_cal_check.clicked.connect(lambda: self.load_cal_file(self.cal_files_cbox.currentText()))
        self.apply_cal_check.clicked.connect(self.update_spec)
        self.cal_files_cbox.currentTextChanged.connect(self.load_cal_file)
        
        
        # line identification
        self.ident_go.clicked.connect(self.ident.update_spec_ident)
        self.ident_table.setColumnWidth(0, 35)
        self.ident_table.setColumnWidth(1, 50)
        self.ident_table.setColumnWidth(2, 60)
        self.ident_table.setColumnWidth(3, 80)
        self.ident_table.setColumnWidth(4, 120)
        self.ident_Te.hide()
        self.ident_Te_label.hide()
        self.ident_int_cbox.currentIndexChanged.connect(self.ident.ident_int_changed)
        self.action_export_ident_table.triggered.connect(self.ident.save_NIST_data)
        self.ident_clear.clicked.connect(self.actionClear_Ident_Plots.triggered)
        self.actionClear_Ident_Plots.triggered.connect(self.ident.clear_spec_ident)

        self.working = 0
        
        # continuum radiation
        self.show_continuum_btn.clicked.connect(self.cont.plot_continuum0)
        self.fit_continuum_btn.clicked.connect(self.cont.fit_continuum)
        self.cont_save_btn.clicked.connect(self.action_export_continuum_fit_results.trigger)
        self.action_export_continuum_fit_results.triggered.connect(self.cont.save_continuum_results)
        self.clear_continuum_btn.clicked.connect(self.actionClear_Continuum_Plots.trigger)
        self.actionClear_Continuum_Plots.triggered.connect(self.cont.clear_continuum)
        self.cont_clear_data_btn.clicked.connect(self.actionClear_Continuum_Table.trigger)
        self.actionClear_Continuum_Table.triggered.connect(self.cont.clear_continuum_table)
        self.cont_fit_results_table.customContextMenuRequested.connect(self.cont_fit_results_rightClick)

        # molecules
        self.mol_multitemp_group.hide()
        self.mol_show_btn.clicked.connect(self.mol.show_spec)
        self.mol_fit_btn.clicked.connect(self.mol.fit)
        self.mol_clear_btn.clicked.connect(self.actionClear_Molecule_Plots.trigger)
        self.actionClear_Molecule_Plots.triggered.connect(self.mol.clear_spec)
        self.mol_save_btn.clicked.connect(self.action_export_molecule_fit_results.trigger)
        self.action_export_molecule_fit_results.triggered.connect(self.mol.save_results)
        self.mol_clear_data_btn.clicked.connect(self.actionClear_Molecule_Table.trigger)
        self.actionClear_Molecule_Table.triggered.connect(self.mol.clear_table)
        self.mol_fit_results_table.customContextMenuRequested.connect(self.mol.fit_results_rightClick)


        # opening/closing of the left/right splitter panes
        self.splitter.splitterMoved.connect(self.fix_view_action_State)
        
        lhandle = self.splitter.handle(1)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.lh_button = QToolButton(lhandle)
        self.lh_button.setFixedSize( 12, 150 ) 
        self.lh_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.lh_button.clicked.connect(self.actionShow_Left_Pane.trigger)
        self.actionShow_Left_Pane.triggered.connect(self.toggle_left_pane)
        layout.addWidget(self.lh_button)
        lhandle.setLayout(layout)

        rhandle = self.splitter.handle(2)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.rh_button = QToolButton(rhandle)
        self.rh_button.setFixedSize( 12, 150 ) 
        self.rh_button.setArrowType(Qt.ArrowType.RightArrow)
        self.rh_button.clicked.connect(self.actionShow_Right_Pane.trigger)
        self.actionShow_Right_Pane.triggered.connect(self.toggle_right_pane)
        layout.addWidget(self.rh_button)
        rhandle.setLayout(layout)

        self.check_HideLegend.checkStateChanged.connect(
            lambda state: self.specplot.plotItem.legend.setVisible(state == Qt.CheckState.Checked)
        )


##############################################################################
# <-------------------- basic/general UI function -------------------------> #
##############################################################################

    def eventFilter(self, source, event): #prevents spec select on right click
        if hasattr(event, 'button'):
            if (event.button() == Qt.RightButton and
                source is self.file_list.viewport()):
                return True
        return super(Window, self).eventFilter(source, event)


    def toggle_left_pane(self):
        w = self.splitter.size().width()
        fac = [8,25,10] # stretch factors. Adjust if changed in designer
        left = self.splitter.widget(0)
        right = self.splitter.widget(2)
        if left.visibleRegion().isEmpty() and not right.visibleRegion().isEmpty():
            self.splitter.setSizes([int(w*fac[0]),int(w*fac[1]),int(w*fac[2])])
            self.lh_button.setArrowType(Qt.ArrowType.LeftArrow)
        elif left.visibleRegion().isEmpty() and right.visibleRegion().isEmpty():    
            self.splitter.setSizes([int(w*fac[0]),int(w*fac[1]),0])
            self.lh_button.setArrowType(Qt.ArrowType.LeftArrow)
        elif not left.visibleRegion().isEmpty() and right.visibleRegion().isEmpty(): 
            self.splitter.setSizes([0,int(w*fac[1]),0])
            self.lh_button.setArrowType(Qt.ArrowType.RightArrow) 
        else:
            self.splitter.setSizes([0,int(w*fac[1]),int(w*fac[2])])
            self.lh_button.setArrowType(Qt.ArrowType.RightArrow) 
            

    def toggle_right_pane(self):
        w = self.splitter.size().width()
        fac = [8,25,10]
        left = self.splitter.widget(0)
        right = self.splitter.widget(2)
        if right.visibleRegion().isEmpty() and not left.visibleRegion().isEmpty():
            self.splitter.setSizes([int(w*fac[0]),int(w*fac[1]),int(w*fac[2])])
            self.rh_button.setArrowType(Qt.ArrowType.RightArrow)
        elif right.visibleRegion().isEmpty() and left.visibleRegion().isEmpty():    
            self.splitter.setSizes([0,int(w*fac[1]),int(w*fac[2])])
            self.rh_button.setArrowType(Qt.ArrowType.RightArrow)
        elif not right.visibleRegion().isEmpty() and left.visibleRegion().isEmpty(): 
            self.splitter.setSizes([0,int(w*fac[1]),0])
            self.rh_button.setArrowType(Qt.ArrowType.LeftArrow) 
        else:
            self.splitter.setSizes([int(w*fac[0]),int(w*fac[1]),0])
            self.rh_button.setArrowType(Qt.ArrowType.LeftArrow)  
            

    def fix_view_action_State(self):
        left = self.splitter.widget(0)
        right = self.splitter.widget(2)
        if left.visibleRegion().isEmpty():
            self.actionShow_Left_Pane.setChecked(False)
        else:
            self.actionShow_Left_Pane.setChecked(True)
            
        if right.visibleRegion().isEmpty():
            self.actionShow_Right_Pane.setChecked(False)
        else:
            self.actionShow_Right_Pane.setChecked(True)

    def file_list_keys(self, event):
        if event.key() == Qt.Key.Key_Delete:
            iterator = QTreeWidgetItemIterator(self.file_list,flags=QTreeWidgetItemIterator.IteratorFlag.Selected)
            root = self.file_list.invisibleRootItem()
            targets = []
            # Don't remove items in iterator since it changes the length/item index positions.
            while iterator.value():
                this_item = iterator.value()
                targets.append(this_item)
                iterator += 1
            for t in targets:
                t.remove()   
        else:
            QTreeWidget.keyPressEvent(self.file_list, event)
        event.accept()
            
    def on_open_bg_file(self):
        """"Open a separate background file (not from the File Tree) via a FileDialog.
        
        This loads data, creates a SpectrumTreeItem and associates it with the currently active spectra.
        The SpectrumTreeItem is not added to the file tree however, and the background spectrum will not be plotted (or be plottable).
        """
        path,filter = QFileDialog.getOpenFileName(caption='Open background file')
        path = Path(path).resolve()
        if path.exists() & path.is_file():
            from OES_toolbox.file_handling import FileLoader
            data = FileLoader.open_any_spectrum(path)
            item = SpectrumTreeItem(path,label='Background')
            if len(data)>1:
                roi_idx, accepted =  QInputDialog.getInt(self,"Pick background ROI","Pick ROI index",min=0,max=len(data)-1)
                if not accepted:
                    return
            else:
                roi_idx=0
            if np.ndim(data[roi_idx].y)>1:
                idx,accepted = QInputDialog.getInt(
                    self,
                    "Pick background spectrum", 
                    f"Pick index of spectrum from file (out of {data[roi_idx].shape[1]})", 
                    min=0,
                    max=data[roi_idx].shape[1]-1
                )
                if not accepted:
                    return
            else:
                idx = 0
            y = data[roi_idx].y if np.ndim(data[roi_idx].y)<2 else data[roi_idx].y[:,idx]
            item.set_spectrum(data[roi_idx].x,y,bg=data[roi_idx].background)
            self.on_set_background_action(item)


##############################################################################
# <------------------------- general plotting -----------------------------> #
##############################################################################

    def update_plot_pos(self, pos):
        pos = self.specplot.getPlotItem().vb.mapSceneToView(pos[0])
        x = f"{pos.x():07.3f}"
        x = f'{x:7.7}'
        y = f"{pos.y():#6.3g}"
        y = f'{y:9.9}'
        y = y.rstrip('. ')

        self.pos_display.setText("   (" + x + ', ' + str(y) + ")")


    def plot(self, x,y, name):
        self.specplot.plot(x=x, y=y, name=name)            
        
    
    def update_spec_colors(self):
        """Walks through the plotted curves and assignes colors."""
        cc = 0   
        for plot_item in self.specplot.listDataItems():
            if "file:" in plot_item.name():
                pen = pg.mkPen(color=colors[cc])
                plot_item.setPen(pen)
                plot_item.setZValue(1)
                cc = cc + 1
                cc = cc%len(colors)
                
        for plot_item in self.specplot.listDataItems():                     
            if "cont.:" in plot_item.name():
                pen = pg.mkPen(color=colors[cc], width=2)
                plot_item.setPen(pen)
                plot_item.setZValue(10)
                cc = cc + 1
                cc = cc%len(colors)   
                
        for plot_item in self.specplot.listDataItems():     
            if "molecule:" in plot_item.name():
                pen = pg.mkPen(color=colors[cc], style=Qt.PenStyle.DashLine)
                plot_item.setPen(pen)
                plot_item.setZValue(20)
                cc = cc + 1
                cc = cc%len(colors)
        
        for plot_item in self.specplot.listDataItems():     
            if "NIST:" in plot_item.name():
                pen = pg.mkPen(color=colors[cc], style=Qt.PenStyle.DashLine, width=1.0)
                plot_item.setPen(pen)
                cc = cc + 1
                cc = cc%len(colors)
    
    
    def update_progress_bar(self,p):
        self.working = self.working + p
        if self.working == 0:
            self.progress_bar.hide()
            self.status_msg.show()
            self.ident_go.setEnabled(True)
            self.ident_clear.setEnabled(True)
        else:
            self.progress_bar.show()
            self.status_msg.hide()
            self.ident_go.setEnabled(False)
            self.ident_clear.setEnabled(False)
            
            
    def graph_to_clipboard(self):
        from OES_toolbox.file_handling import FileExport
        fig = FileExport.graph_to_matplotlib(self.specplot)
        img = FileExport.matplotlib_to_image(fig)
        QApplication.clipboard().setImage(img)
            

##############################################################################
# <-------------------- Plotting measurement data -------------------------> #
##############################################################################
    
    def on_current_item_changed(self,current:SpectrumTreeItem,previous:SpectrumTreeItem):
        if (previous is not None) and isinstance(previous._external_bg,SpectrumTreeItem):
            previous._external_bg.setIcon(0,previous._external_bg._ICON_FILE_CACHED if previous._external_bg.is_file_node_item else QtGui.QIcon())
            previous._external_bg.setStatusTip(0,None)
        if current is not None:
            if isinstance(current._external_bg, SpectrumTreeItem):
                bg_path = current._external_bg.path.as_posix()
                current._external_bg.setIcon(0,current._ICON_BG)
                current._external_bg.setStatusTip(0,"Active background spectrum")
            else:
                bg_path = ""



    def plot_filetree_item(self, this_item:SpectrumTreeItem):
        """Loads file and plots content."""
        self.logger.debug(f"{this_item.label}: {this_item.is_loaded=}")
        if (not this_item.is_loaded) & (this_item.is_file):
            try:
                this_item.load_data()
            except (AttributeError,UnboundLocalError, EncodingWarning):
                self.status_msg.setText(f"Could not load data from {this_item.path.name}")
                return
            self.status_msg.setText(f"Loading file {this_item.path.name} complete!")
        this_item.add_to_graph(self.specplot)
                
    def update_spec(self):
        """Checks which files are selected for plotting, loads and plots them."""
        update_on_selected = self.plot_combobox.currentIndex() == 0
        sender = self.sender()
        sender_text = sender.text() if hasattr(sender,'text') else None
        self.logger.warning(f"`Update spec` called by {sender} {sender_text=}; {update_on_selected=}")

        if update_on_selected:
            self.on_selection_change()
        else:
            iterator = QTreeWidgetItemIterator(
                self.file_list,
            )  # flags=QTreeWidgetItemIterator.IteratorFlag.Checked)
            while iterator.value():
                item = iterator.value()
                iterator += 1
                self.on_check_change(item, 0)


    def update_file_info_box(self, selected):
        if len(selected) == 1:
            this_item = selected[0]
            self.spec_info_gbox.setEnabled(True)
            self.sel_spec_label.setText(this_item.name(shorten=True))
            self.reload_file_btn.setEnabled(this_item.is_file_node_item)
            self.clear_file_btn.setEnabled(this_item.is_file_node_item)
            bg_item = this_item._external_bg

            if isinstance(bg_item,SpectrumTreeItem):
                bg_label = bg_item.name()
                self.bg_extra_ledit.setText(bg_label)
            else:
                self.bg_extra_ledit.setText("")

            self.bg_extra_check.setChecked(isinstance(bg_item,SpectrumTreeItem))
            self.bg_extra_check.setEnabled(isinstance(bg_item,SpectrumTreeItem))
        
        else:
            self.spec_info_gbox.setEnabled(False)
            self.sel_spec_label.setText(" ")


    def on_selection_change(self):
        update_on_selected = self.plot_combobox.currentIndex() == 0
        self.logger.debug(f"Selection Changed -> {update_on_selected=}")
        selected = self.file_list.selectedItems()

        if update_on_selected:
            viewbox = self.specplot.getViewBox()
            autorange_state:list[bool] = viewbox.getState()['autoRange']
            autorange_flag: bool = True in autorange_state
            if autorange_flag:
                viewbox.disableAutoRange()
            iterator = QTreeWidgetItemIterator(self.file_list, flags=QTreeWidgetItemIterator.IteratorFlag.Unselected)
            while iterator.value():
                this_item:SpectrumTreeItem = iterator.value()
                iterator += 1
                if not this_item._is_selected_with_ancestors():
                    this_item.remove_from_graph(self.specplot)
            for this_item in selected:
                if not this_item.is_dir:
                    self.plot_filetree_item(this_item)
            if autorange_flag:
                viewbox.autoRange()
                self.logger.debug(f"Autoranging-> {autorange_state=}")
            viewbox.enableAutoRange(x=autorange_state[0],y= autorange_state[1])
        self.update_file_info_box(selected)
        self.update_spec_colors()


    def on_bg_check_change(self, checked):
        if not checked:
              self.on_set_background_action(None)


    def on_check_change(self, item, col):
        update_on_check = self.plot_combobox.currentIndex() == 1
        if update_on_check:
            viewbox = self.specplot.getViewBox()
            autorange_state:list[bool] = viewbox.getState()['autoRange']
            autorange_flag: bool = True in autorange_state
            if autorange_flag:
                viewbox.disableAutoRange()
            check_state: bool = item.checkState(col) == Qt.CheckState.Checked
            if check_state | item._is_checked_with_ancestors():
                self.plot_filetree_item(item)
            else:
                item.remove_from_graph(self.specplot)
                # persist checked children
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.checkState(col) == Qt.CheckState.Checked:
                        child.add_to_graph(self.specplot)
            if autorange_flag:
                viewbox.autoRange()
                self.logger.debug(f"Autoranging-> {autorange_state=}")
            viewbox.enableAutoRange(x=autorange_state[0],y= autorange_state[1])
            self.update_spec_colors()
    

    def clear_all_spec(self):
        self.mol.clear_spec()
        self.ident.clear_spec_ident()
        self.cont.clear_continuum()
        self.file_list.clearSelection()
        iterator = QTreeWidgetItemIterator(self.file_list,flags=QTreeWidgetItemIterator.IteratorFlag.Checked)
        while iterator.value():
            iterator.value().setCheckState(0,Qt.CheckState.Unchecked)
            iterator += 1
            

##############################################################################
# <--------------------------- drag & drop --------------------------------> #
##############################################################################

    def check_drag_drop(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if not os.path.exists(path):
                    event.reject()
                
            event.accept()
            
        else:
            event.ignore()
        
        
    def do_drag_drop(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            item = self.filetree_item(path)
            item.iterdir()
            self.file_list.addTopLevelItem(item)
        event.accept()


    def filetree_item(self, path, is_content=False, num=0, label="Scan"): # TODO: should really be a class
        path = Path(path)
        item = SpectrumTreeItem(path = path, label=f"{label} {num}", content_num=None, is_content=is_content)
        return item
    

##############################################################################
# <--------------------------- right click --------------------------------> #
##############################################################################

    def file_rightClick(self, cursor):
        file_item: SpectrumTreeItem = self.file_list.itemAt(cursor)
        if file_item is None:
            return

        menu = QMenu()
        
        reload_action = QAction(qta.icon("mdi6.reload"),"Reload this file")
        bg_action = QAction(qta.icon("mdi.layers"),"Use as background", checkable=True)
        reset_bg_action = QAction(qta.icon("mdi.layers-off",color=("red",200)),"Reset background")
        del_this_action = QAction("Clear this item")
        del_selected_action = QAction("Clear selected")
        del_unselected_action = QAction("Clear not selected")
        del_unchecked_action = QAction("Clear not checked")
        clear_action = QAction("Clear all")
        
        if file_item.is_file_node_item:
            menu.addAction(reload_action)
            reload_action.triggered.connect(lambda:self.on_reload_file_action(file_item))   
        # TODO: implement toggling of external background
        bg_atm = (file_item==file_item._external_bg) #& self.bg_extra_check.isChecked()
    
        bg_action.setChecked(bg_atm)

        if (file_item.childCount() == 0) & (not file_item.is_dir):
            menu.addAction(bg_action)
        if (file_item.is_file_node_item or file_item.is_content or 
                                isinstance(file_item._external_bg ,SpectrumTreeItem)):
            menu.addAction(reset_bg_action)
        
        menu_clear = menu.addMenu("Clear")
        menu_clear.addAction(del_this_action)
        menu_clear.addAction(del_selected_action)
        menu_clear.addAction(del_unselected_action)
        menu_clear.addAction(del_unchecked_action)
        menu_clear.addAction(clear_action)

        bg_action.triggered.connect(lambda: self.on_set_background_action(file_item))
        clear_action.triggered.connect(self.on_file_clear_action)
        del_this_action.triggered.connect(file_item.remove)
        del_selected_action.triggered.connect(self.on_file_clear_action)
        del_unselected_action.triggered.connect(self.on_file_clear_action)
        del_unchecked_action.triggered.connect(self.on_file_clear_action)
        reset_bg_action.triggered.connect(lambda: self.on_set_background_action(None))

        menu.exec(QtGui.QCursor.pos())

    def on_set_background_action(self,item:SpectrumTreeItem):
        update_on_selected = self.plot_combobox.currentIndex() == 0
        flag = QTreeWidgetItemIterator.IteratorFlag.Selected if update_on_selected else QTreeWidgetItemIterator.IteratorFlag.Checked
        iterator =  QTreeWidgetItemIterator(self.file_list,flag)
        while iterator.value():
            some_item:SpectrumTreeItem = iterator.value()
            some_item.set_background(item)
            iterator += 1
        self.bg_extra_ledit.setText("" if item is None else item.name())
        self.bg_extra_check.setChecked(isinstance(item,SpectrumTreeItem))
        self.bg_extra_check.setEnabled(isinstance(item,SpectrumTreeItem))


    def on_file_clear_action(self,*args): 
        """"Handle clearing (a subset of) files and spectra in response to an action."""
        triggered_by = self.sender().text()
        self.logger.info(f"Action fired: {triggered_by}")
        match triggered_by:
            case "Clear selected"|"Clear file":
                targets: list[SpectrumTreeItem]  = self.file_list.selectedItems()
            case "Clear not selected":
                targets: list[SpectrumTreeItem] = []
                iterator = QTreeWidgetItemIterator(self.file_list,QTreeWidgetItemIterator.IteratorFlag.Unselected)
                while iterator.value():
                    item:SpectrumTreeItem = iterator.value()
                    is_selected = item._is_selected_with_descendants() | item._is_selected_with_ancestors()
                    if not is_selected:
                        targets.append(item)
                    iterator += 1
            case "Clear not checked":
                targets: list[SpectrumTreeItem] = []
                iterator = QTreeWidgetItemIterator(self.file_list,QTreeWidgetItemIterator.IteratorFlag.NotChecked)
                while iterator.value():
                    item:SpectrumTreeItem = iterator.value()
                    is_checked = item._is_checked_with_descendants() | item._is_checked_with_ancestors()
                    if not is_checked:
                        targets.append(item)
                    iterator += 1
            case "Clear all" | "Clear Files":
                targets: list[SpectrumTreeItem] = [self.file_list.topLevelItem(i) for i in range(self.file_list.topLevelItemCount())]
            case _:
                targets = None
        for item in targets:
            current_index = self.file_list.indexFromItem(item)
            if current_index.isValid():
                self.file_list.itemFromIndex(current_index).remove()

    def on_reload_file_action(self, file_item:SpectrumTreeItem):
        was_plotted = file_item.is_plotted(self.specplot)
        was_selected = (file_item in self.file_list.selectedItems())
        file_item.is_loaded = False
        file_item.clear_children()
        if was_plotted:
            self.plot_filetree_item(file_item)
        else:
            if (not file_item.is_loaded) & (file_item.is_file):
                try:
                    file_item.load_data()
                except (AttributeError,UnboundLocalError):
                    self.status_msg.setText(f"Could not load data from {file_item.path.name}")
                    return
                self.status_msg.setText(f"Loading file {file_item.path.name} complete!")
        if was_selected and not was_plotted:
            self.on_selection_change()
        if file_item.childCount()==0:
            file_item.set_background(None)
        self.update_spec_colors()
        self.bg_extra_ledit.setText("")
        self.bg_extra_check.setChecked(False)
        self.bg_extra_check.setEnabled(False)


    def cont_fit_results_rightClick(self, cursor):
        row = self.cont_fit_results_table.rowAt(cursor.y())
        # col = self.mol_fit_results_table.columnAt(cursor.x())
        plot_label = self.cont_fit_results_table.item(row, 0).plot_label

        menu = QMenu()
        plot_action = QAction("Plot row", checkable=True)
        del_row_action = QAction("Remove row")
        clear_action = QAction("Clear table")
        
        plotted_atm = False
        for plot_item in self.specplot.listDataItems():
            if "cont.: " + plot_label in plot_item.name():
                plotted_atm = True

        if plotted_atm:
            plot_action.setChecked(True)
        else:
            plot_action.setChecked(False)

        menu.addAction(plot_action)
        menu.addAction(del_row_action)
        menu.addAction(clear_action)

        plot_action.triggered.connect(lambda: self.cont.plot_cont_table_item(row, plot_action.isChecked()))
        del_row_action.triggered.connect(lambda: self.cont.del_continuum_table_row(row))
        del_row_action.triggered.connect(lambda: self.cont.plot_cont_table_item(row, False))
        clear_action.triggered.connect(self.cont.clear_continuum_table)

        menu.exec(QtGui.QCursor.pos())


##############################################################################
# <----------------------- spectromter setting ----------------------------> #
##############################################################################

    def open_cal_folder(self):
        """Open folder with calibration files with OS-native file explorer"""
        match platform.system():
            case "Windows":
                os.startfile(self.cal_path)
            case "Darwin":
                subprocess.call(["open",Path(self.cal_path).resolve().as_posix()])
            case _:
                subprocess.call(["xdg-open",Path(self.cal_path).resolve().as_posix()])


    def add_cal_file(self):
        cal_file, _ = QFileDialog.getOpenFileName(caption='Open calibration file')
        if cal_file is not None:
            target = Path(self.cal_path).joinpath(Path(cal_file).name).resolve()
            already_exists = target.exists()
            if already_exists:
                picked = QMessageBox.question(
                    self,
                    "Calibration file already exists",
                    f"A calibration named '{target.name}' already exists, overwrite?",
                    buttons = QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
                    )
                if picked == QMessageBox.StandardButton.No:
                    return
            # make sure the file is valid, by loading it to test it
            tmp=FileLoader._read_generic_text(cal_file)
            if np.shape(tmp)[1]!=2:
                QMessageBox.warning(self,*INVALID_CALIB_TXT,QMessageBox.StandardButton.Ok)
                return
            QFile.copy(cal_file, target.as_posix())
            self.cal_files_cbox.addItem(target.name)
            self.cal_files_cbox.setCurrentText(target.name)


    def cal_files_refresh(self):      
        """Refresh the list of calibrations, adding newly added files and removing item for files that are no longer there.
        
        The currently selected item is kept active (if it remains valid), which avoid firing a currentTextChanged signal.
        """  
        files = sorted([f.name for f in Path(self.cal_path).resolve().iterdir() if f.is_file()])
        # currentChoice = self.cal_files_cbox.currentText()
        currentItems = [self.cal_files_cbox.itemText(i) for i in range(self.cal_files_cbox.count())]
        to_remove = [i for i,elem in enumerate(currentItems) if elem not in files][::-1]
        to_add = [elem for elem in files if elem not in currentItems]
        for elem in to_remove:
            self.cal_files_cbox.removeItem(elem)
        self.cal_files_cbox.addItems(to_add)


    def load_cal_file(self, filename):
        """ Tests validity of cal file by loading it. Might as well already 
        load it and save it to self.cal, if we test-load it anyway..."""
        if len(filename) > 0:
            try:
                calib = FileLoader._read_generic_text(Path(self.cal_path).joinpath(filename))
                if calib.shape[1]!=2:
                    raise ValueError(f"Calibration file {filename} should only have two columns, got: {calib.shape[1]}")
                # `interp1d` is deprecated; use modern API instead, which creates a callable BSpline instance, with k=1 for linear interpolation
                self.cal = scipy.interpolate.make_interp_spline(calib.iloc[:,0], calib.iloc[:,1],k=1) 
                self.apply_cal_check.setEnabled(True)
            except:
                self.apply_cal_check.setEnabled(False)
                QMessageBox.warning(self,*INVALID_CALIB_TXT,QMessageBox.StandardButton.Ok)
        else:
            self.apply_cal_check.setEnabled(False)


    def cal_info_text(self, index):
            if index.isValid():
                QtGui.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    index.data(),
                    self.view.viewport(),
                    self.view.visualRect(index)
                    )
    
    

def run(app, splash):
    app.setApplicationName("OES toolbox")

    win = Window()
    win.show()
    splash.finish(win)
    sys.exit(app.exec())
