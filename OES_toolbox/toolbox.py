import os
import sys
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QFileDialog, QTreeWidgetItem, \
        QTreeWidgetItemIterator , QHeaderView, \
        QMainWindow, QVBoxLayout, QToolButton, QDialog, \
        QDialogButtonBox, QLabel, QMenu
from PyQt6.QtCore import Qt, QSettings, \
        QStandardPaths, QFile
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6 import sip, QtGui
import pyqtgraph

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
file_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(file_dir + '/ui/jh-paper.mplstyle')

from .ui import resources # seems unused but is needed!
from OES_toolbox.settings import settings
from OES_toolbox.io import fio
from OES_toolbox.ident import ident_module
from OES_toolbox.molecules import molecule_module
from OES_toolbox.continuum import cont_module



colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # matplotlib default
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')
pyqtgraph.setConfigOptions(antialias=True)



class cal_invalid_dialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Invalid calibration file format")

        msg = """Invalid or unreadable calibration file. 
Please make sure that the file is a text file which contains only two columns, 
with the wavelength in nm and the sensitifity of the spectrometer in arbitrary 
units, representing the photon count per second. The file must use a point as 
the decimal delimiter and tab as the delimiter between the two columns."""

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
        self.progress_bar.hide()
        
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
        self.cal = [[],[]]
        self.cal_files_refresh()
        
        # center plot
        self.specplot.setLabel("left", "intensity")
        self.specplot.setLabel("bottom", "wavelength / nm")
        self.specplot.addLegend()
        self.wl_shift.valueChanged.connect(self.update_spec)
        self.copy_plots_btn.clicked.connect(self.action_graph_to_clipboard.trigger)
        self.action_graph_to_clipboard.triggered.connect(self.graph_to_clipboard)
        self.action_export_plot_data.triggered.connect(self.io.save_plots)
        self.actionRefresh_plots.triggered.connect(self.update_spec)
        self.proxy = pyqtgraph.SignalProxy(self.specplot.scene().sigMouseMoved, rateLimit=90, slot=self.update_plot_pos)
        # self.actionClear_Plots.triggered.connect() # TODO clear all function! Needs to touch a lot of things
    
        # file loading, plotting /drag & drop
        self.file_opt_group.hide() # TODO not a thing yet, hide for now
        self.button_open.clicked.connect(self.actionOpenFolder.trigger)
        self.button_open_files.clicked.connect(self.actionOpenFiles.trigger)
        self.actionOpenFolder.triggered.connect(self.io.open_folder)
        self.actionOpenFiles.triggered.connect(self.io.open_files)
        
        self.file_list.itemChanged.connect(self.update_spec_checkbox) # checkmark
        self.file_list.itemSelectionChanged.connect(self.update_spec_selected) # highlight
        self.plot_combobox.currentIndexChanged.connect(self.update_spec)
        self.file_list.dropEvent = self.do_drag_drop
        self.file_list.dragEnterEvent = self.check_drag_drop
        self.file_list.dragMoveEvent = self.check_drag_drop
        self.file_list.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.clear_file_list.clicked.connect(self.actionClearFiles.trigger)
        self.actionClearFiles.triggered.connect(self.file_list.clear)
        self.actionClearFiles.triggered.connect(self.update_spec)
        self.file_list.keyPressEvent = self.io.file_list_keys
        self.bg_internal_check.hide()
        self.bg_internal_check.stateChanged.connect(self.update_spec)
        self.bg_extra_btn.clicked.connect(self.io.open_bg_file)
        self.bg_extra_check.stateChanged.connect(self.update_spec)
        self.bg_extra_ledit.num = 0
        self.file_list.customContextMenuRequested.connect(self.file_rightClick)

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


##############################################################################
# <-------------------- basic/general UI function -------------------------> #
##############################################################################

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
            

##############################################################################
# <------------------------- general plotting -----------------------------> #
##############################################################################

    def update_plot_pos(self, pos):
        pos = self.specplot.getPlotItem().vb.mapSceneToView(pos[0])
        x = "{number:07.3f}".format(number=pos.x())
        x = '{:7.7}'.format(x)
        y = "{number:#6.3g}".format(number=pos.y())
        y = '{:9.9}'.format(y)
        y = y.rstrip('. ')

        self.pos_display.setText("   (" + x + ', ' + str(y) + ")")


    def plot(self, x,y, name):
        self.specplot.plot(x=x, y=y, name=name)            
        
    
    def update_spec_colors(self):
        " Walks through the plotted curves and assignes colors."
        cc = 0   
        for plot_item in self.specplot.listDataItems():
            if "file:" in plot_item.name():
                pen = pyqtgraph.mkPen(color=colors[cc])
                plot_item.setPen(pen)
                plot_item.setZValue(1)
                cc = cc + 1
                cc = cc%len(colors)
                
        for plot_item in self.specplot.listDataItems():                     
            if "cont.:" in plot_item.name():
                pen = pyqtgraph.mkPen(color=colors[cc], width=2)
                plot_item.setPen(pen)
                plot_item.setZValue(10)
                cc = cc + 1
                cc = cc%len(colors)   
                
        for plot_item in self.specplot.listDataItems():     
            if "molecule:" in plot_item.name():
                pen = pyqtgraph.mkPen(color=colors[cc], style=Qt.PenStyle.DashLine)
                plot_item.setPen(pen)
                plot_item.setZValue(20)
                cc = cc + 1
                cc = cc%len(colors)
        
        for plot_item in self.specplot.listDataItems():     
            if "NIST:" in plot_item.name():
                pen = pyqtgraph.mkPen(color=colors[cc], style=Qt.PenStyle.DashLine, width=1.0)
                plot_item.setPen(pen)
                cc = cc + 1
                cc = cc%len(colors)
    
    
    def update_progress_bar(self,p):
        self.working = self.working + p
        if self.working == 0:
            self.progress_bar.hide()
            self.ident_go.setEnabled(True)
            self.ident_clear.setEnabled(True)
        else:
            self.progress_bar.show()
            self.ident_go.setEnabled(False)
            self.ident_clear.setEnabled(False)
            
            
    def graph_to_clipboard(self):
        xlim, ylim = self.specplot.getViewBox().viewRange()
        fig = plt.figure()
        # fig = plt.gca()
        canvas = FigureCanvas(fig)
        style = ''
        for plot_item in self.specplot.listDataItems():
            if "file" in plot_item.name():
                x,y = plot_item.getData()          
                plt.plot(x,y, style, label=plot_item.name(), zorder=3)

        for plot_item in self.specplot.listDataItems():                     
            if "cont" in plot_item.name():
                x,y = plot_item.getData()          
                plt.plot(x,y, style, label=plot_item.name())

        for plot_item in self.specplot.listDataItems():     
            if "molecule" in plot_item.name():
                x,y = plot_item.getData() 
                style = '--'
                plt.plot(x,y, style, label=plot_item.name())
        
        for plot_item in self.specplot.listDataItems():     
            if "NIST" in plot_item.name():
                x,y = plot_item.getData()   
                style = ':'
                plt.plot(x,y, style, label=plot_item.name(), lw=0.5)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('wavelength / nm')
        plt.ylabel('intensity')
        canvas.draw()

        width, height = fig.figbbox.width, fig.figbbox.height
        img = QImage(canvas.buffer_rgba(), int(width), int(height), QImage.Format.Format_RGBA8888_Premultiplied)
        pixmap = QPixmap(img)

        QApplication.clipboard().setImage(pixmap.toImage())
            

##############################################################################
# <-------------------- Plotting measurement data -------------------------> #
##############################################################################

    def update_spec_selected(self):
        if self.plot_combobox.currentIndex() == 0: # plot selected
            self.update_spec()
        
        
    def update_spec_checkbox(self):
        if self.plot_combobox.currentIndex() == 1:  # plot checked
            self.update_spec()
       
    
    def plot_filetree_item(self, this_item):
        """ Loads file and plots content. Walks up the tree to assemble the path. """
        path = []
        num = 0
        if this_item.is_content:
            current_parent =  this_item.parent().parent()
            path.append(this_item.parent().text(0))
            num = this_item.content_num
            label = "file: " + this_item.parent().text(0) + ", Scan " + str(num+1)
        else:
            current_parent = this_item.parent()
            path.append(this_item.text(0))
            label = "file: " + this_item.text(0)
        
        while not current_parent is None:
            path.append(current_parent.text(0))
            current_item = current_parent
            current_parent = current_item.parent()
            
        path = os.path.join(self.io.active_folder, *path[::-1])
        x,y = self.io.open_file(path, this_item, num=num, content=this_item.is_content)
        if x is None:
            x = np.arange(0, np.shape(y)[-1])
            
        if np.ndim(y) > 1 and np.shape(y)[0] > 1:
            for i,this_y in enumerate(y):
                label = "file: " + this_item.text(0) + ", Scan " + str(i+1)
                self.plot(x+self.wl_shift.value(), this_y, name=label)
        elif np.ndim(y) > 1:
            self.plot(x+self.wl_shift.value(), y[0], name=label)
        else:
            self.plot(x+self.wl_shift.value(), y, name=label)

    
    def plot_children(self,item):
        """ Recursivly walks through all children of selected tree item. Calls
        plot_filetree_item for each leaf. """
        if item.childCount() == 0:
            self.plot_filetree_item(item)
        else:
            for idx in range(item.childCount()):
                child = item.child(idx)
                self.plot_children(child)
                
                
    def update_spec(self):
        """ Checks which files are selected for plotting, loads and plots them. """
        # remove file specs
        for plot_item in self.specplot.listDataItems():
            if "file" in plot_item.name():
                self.specplot.removeItem(plot_item)
        
        # add file specs
        iterator = QTreeWidgetItemIterator(self.file_list)
        while iterator.value():
            this_item = iterator.value()
            iterator += 1
            if (self.plot_combobox.currentIndex() == 0 and this_item.isSelected()) \
                or (self.plot_combobox.currentIndex() == 1 
                    and this_item.checkState(0) == Qt.CheckState.Checked):
            
                self.plot_children(this_item)
                    
        self.update_spec_colors()
                 
            
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
            path = url.toLocalFile()
            item = self.filetree_item(path)
            self.file_list.addTopLevelItem(item)
            if os.path.isdir(path):
                self.io.add_sub(path, item)
        event.accept()

 
    def filetree_item(self, label, is_content=False, num=0): # TODO: should really be a class
        item = QTreeWidgetItem()
        item.setText(0, label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0,Qt.CheckState.Unchecked)
        item.setBackground(0, QtGui.QColor('white'))
        item.is_content = is_content
        item.content_num = num
        return item
    

##############################################################################
# <--------------------------- right click --------------------------------> #
##############################################################################

    def file_rightClick(self, cursor):
        file_item = self.file_list.itemAt(cursor)
        num = 0
        path = []
        if file_item.is_content:
            current_parent =  file_item.parent().parent()
            path.append(file_item.parent().text(0))
            num = file_item.content_num
        else:
            current_parent = file_item.parent()
            path.append(file_item.text(0))
            
        while not current_parent is None:
            path.append(current_parent.text(0))
            current_item = current_parent
            current_parent = current_item.parent()
        path = os.path.join(self.io.active_folder, *path[::-1])

        menu = QMenu()
        bg_action = QAction("Use as background", checkable=True)
        del_file_action = QAction("Clear selected")
        clear_action = QAction("Clear all")
        
        bg_atm = False
        if self.bg_extra_ledit.text() == path and self.bg_extra_check.isChecked():
            if not file_item.is_content:
                bg_atm = True
            else:
                if self.bg_extra_ledit.num == file_item.content_num:
                    bg_atm = True

        if bg_atm:
            bg_action.setChecked(True)
        else:
            bg_action.setChecked(False)

        if file_item.childCount() == 0:
            menu.addAction(bg_action)
            
        if not file_item.is_content:
            menu.addAction(del_file_action)
        menu.addAction(clear_action)

        bg_action.triggered.connect(lambda: self.file_rightclick_bg_action(path, bg_action.isChecked(), num))
        del_file_action.triggered.connect(lambda: sip.delete(file_item))
        del_file_action.triggered.connect(self.update_spec)
        clear_action.triggered.connect(self.file_list.clear)

        menu.exec(QtGui.QCursor.pos())
        
    
    def file_rightclick_bg_action(self, bg_path, enabled, num):
        if enabled:
            self.bg_extra_ledit.setText(bg_path)
            self.bg_extra_ledit.num = num
            self.bg_extra_check.setChecked(True)
        else:
            self.bg_extra_check.setChecked(False)
        self.update_spec()


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
        os.startfile(self.cal_path)


    def add_cal_file(self):
        cal_file, _ = QFileDialog.getOpenFileName(caption='Open calibration file')
        filename = os.path.basename(cal_file)
        QFile.copy(cal_file, os.path.join(self.cal_path, filename))


    def cal_files_refresh(self):
        # block signal so that test_cal_file doesn't fire on an empty string
        self.cal_files_cbox.blockSignals(True) 
        
        files = sorted(os.listdir(self.cal_path))
        self.cal_files_cbox.clear()
        for f in files:
            self.cal_files_cbox.addItem(f)
        
        self.load_cal_file(self.cal_files_cbox.currentText())
        self.cal_files_cbox.blockSignals(False) # re-enable signals


    def load_cal_file(self, filename):
        """ Tests validity of cal file by loading it. Might as well already 
        load it and save it to self.cal, if we test-load it anyway..."""
        if len(filename) > 0:
            try:
                x,y = np.loadtxt(os.path.join(self.cal_path, filename)).T
                self.cal = interp1d(x, y, bounds_error=False, fill_value=0)
                self.apply_cal_check.setEnabled(True)
            except:
                self.apply_cal_check.setEnabled(False)
                dialog = cal_invalid_dialog()
                dialog.exec()
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
    
    

def run():
    app = QApplication(sys.argv)
    app.setApplicationName("OES toolbox")
    win = Window()
    win.show()
    sys.exit(app.exec())
    