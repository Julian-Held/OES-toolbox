import os
import datetime
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItemIterator, QTableWidgetItem, \
        QMessageBox, QCheckBox, QMenu
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6 import QtGui

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

file_dir = os.path.dirname(os.path.abspath(__file__))

import Moose # free re-implementation of MassiveOES

def model_for_fit(x, T_rot, T_vib, sim_db, instr, resolution=1000, wl_pad=10):
    """Function copied from Moose without the normalization to the maximum. """
    sticks = Moose.create_stick_spectrum(T_vib, T_rot, sim_db)
    refined = Moose.equidistant_mesh(sticks, wl_pad=wl_pad, resolution=resolution)
    simulation = apply_voigt(refined, instr)
    sim_matched = match_spectra(x.reshape(-1, 1), simulation)
    return sim_matched[:, 1]

def apply_voigt(sim, instr):
    """Function copied from Moose to allow arbitrary instrumental functions."""
    x = sim[:, 0]
    conv = fftconvolve(sim[:, 1], instr(x), mode="same")
    return np.array([x, conv]).T

def match_spectra(meas, sim):
    """Function copied from Moose to solve out of bounds errors."""
    interp = interp1d(sim[:, 0], sim[:, 1], bounds_error=False, fill_value=0)
    matched_y = interp(meas[:, 0])
    return np.array([meas[:, 0], matched_y]).T

def get_mOES_spec(x, Tvib, Trot, molecule, instr):
    try:
        sim_y = model_for_fit(x, Trot, Tvib, molecule.db, instr)
    except Exception as e:
        print(e)
        print("Warning: Molecular spectrum not in range?")
        return np.zeros(len(x))
    
    return sim_y/np.sum(sim_y)


class MoleculeFitter(QObject):
    def __init__(self, label, x, y, p0, molecules, sep_Trot, sep_Tvib, instr_func, 
                                            shift=False, stretch=False, parent=None):
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.x, self.y = x,y
        self.p0 = p0
        self.molecules = molecules
        self.sep_Trot, self.sep_Tvib = sep_Trot, sep_Tvib
        self.get_instr = instr_func # function copy from Window class
        self.stop = False # stop flag invoked by button press
        self.shift, self.stretch = shift, stretch

    finished = pyqtSignal()
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)
    # data_ready = pyqtSignal(str, astropy.table.table.Table)
    progress = pyqtSignal(int)


    def fitfunc(self, x, *args):
        p0 = list(args) # needed for pop
        y0 = p0.pop(0)
        stretch, shift = 0, 0
        if self.stretch:
            stretch = p0.pop(-1)
        if self.shift:
            shift = p0.pop(-1)

        if not self.sep_Trot:
            Trot = p0.pop(0)
        if not self.sep_Tvib:
            Tvib = p0.pop(0)

        specs = []
        for mol_sel in self.molecules:
            if mol_sel.isChecked() and mol_sel.can_fit == True:
                A = p0.pop(0)        
                if self.sep_Trot:
                    Trot = p0.pop(0)
                if self.sep_Tvib:
                    Tvib = p0.pop(0)
                x_new = np.mean(x) + ((x - np.mean(x)) * (1 + stretch)) + shift
                this_spec = A*get_mOES_spec(x_new, Tvib, Trot, mol_sel, self.get_instr)
                specs.append(this_spec)
        
        return np.sum(specs, axis=0) + y0


    def fit(self):
        self.progress.emit(1)

        # A and the temps must be >0
        self.bounds = [list(np.zeros(len(self.p0))),
                       list(np.ones(len(self.p0))*np.inf)] 
        self.bounds[0][0] = -np.inf # y0 -> unbound

        if self.shift:
            self.p0.append(0.0)
            self.bounds[0].append(-10) # shift -> +-10
            self.bounds[1].append(10)

        if self.stretch:
            self.p0.append(0.0)
            self.bounds[0].append(-1) # stretch -> +-1
            self.bounds[1].append(1)

        try:
            ans, err = curve_fit(self.fitfunc, self.x, self.y, 
                                    p0=self.p0, bounds=self.bounds)
            y_fit = self.fitfunc(self.x, *ans)

        except Exception as e:
            print(e)
            ans = np.array(self.p0)
            # y_fit = np.zeros(len(self.x))
            y_fit = self.fitfunc(self.x, *ans)
            
        self.result_ready.emit(self.label, ans, self.x, y_fit)
        self.progress.emit(-1)
        self.finished.emit()

    

class molecule_selector(QCheckBox):
    def __init__(self, molecule):
        super().__init__()
        self.label = molecule['label']
        self.setText(self.label)
        self.ident = molecule['ident']
        self.src = molecule['src']
        self.can_fit = False
        if self.src == "mOES":
            self.can_fit = True
    
    def load_db(self, wl=(0,99999)):
        if self.can_fit:
            if self.src == "mOES":
                try:
                    self.db = Moose.query_DB(self.ident, wl=wl)
                except:
                    print("Could not open database for molecular fit.")


        # self.addWidget(QLabel("Channel " + str(self.number)))
        # self.chan_cbox = QComboBox()


##############################################################################
# <------------------------- molecules module -----------------------------> #
##############################################################################   
class molecule_module():
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.get_instr = self.mw.settings.get_instr 

        molecule_list = [{'ident': 'C2_swan', 'label': "C₂ Swan", 'src': 'mOES'},
                         {'ident': 'N2CB', 'label': "N₂ (C-B)", 'src': 'mOES'},
                         {'ident': 'N2PlusBX', 'label': "N₂⁺ (B-X)", 'src': 'mOES'},
                         {'ident': 'NHAX', 'label': "NH (A-X)", 'src': 'mOES'},
                         {'ident': 'NOBX', 'label': "NO (B-X)", 'src': 'mOES'},
                         {'ident': 'OHAX', 'label': "OH (A-X)", 'src': 'mOES'},
                         {'ident': 'NOAX', 'label': "NO (A-X)", 'src': 'LIFBASE'},
                         {'ident': 'NOCX', 'label': "NO (C-X)", 'src': 'LIFBASE'},
                         {'ident': 'NODX', 'label': "NO (D-X)", 'src': 'LIFBASE'},
                         {'ident': 'CHAX', 'label': "CH (A-X)", 'src': 'LIFBASE'},
                         {'ident': 'CHBX', 'label': "CH (B-X)", 'src': 'LIFBASE'},
                         {'ident': 'CHCX', 'label': "CH (C-X)", 'src': 'LIFBASE'},
                         {'ident': 'SiHAX', 'label': "SiH (A-X)", 'src': 'LIFBASE'},
                         {'ident': 'CFAX', 'label': "CF (A-X)", 'src': 'LIFBASE'},
                         {'ident': 'CFBX', 'label': "CF (B-X)", 'src': 'LIFBASE'},
                         {'ident': 'CFCX', 'label': "CF (C-X)", 'src': 'LIFBASE'},
                         {'ident': 'CFDX', 'label': "CF (D-X)", 'src': 'LIFBASE'},
                         {'ident': 'CNBX', 'label': "CN (B-X)", 'src': 'LIFBASE'},
                        ]
            
        self.molecule_selectors = []
        col, row = 0, 0   
        col2, row2 = 0, 0         
      
        for molecule in molecule_list:
            this_mol_check = molecule_selector(molecule)
            if this_mol_check.can_fit:
                this_mol_check.stateChanged.connect(self.change_sel)
                self.molecule_selectors.append(this_mol_check)
                self.mw.mol_select_grid.addWidget(this_mol_check, row, col)
                col = col + 1
                if col == 3:
                    row = row + 1
                    col = 0
            else:
                this_mol_check.stateChanged.connect(self.change_sel)
                self.molecule_selectors.append(this_mol_check)
                self.mw.mol_select_grid_nofit.addWidget(this_mol_check, row2, col2)
                col2 = col2 + 1
                if col2 == 3:
                    row2 = row2 + 1
                    col2 = 0
                
        self.mol_fit_threads = []
        self.mol_fit_workers = []
        

    def fitfunc(self, x, *args):
        p0 = list(args) # needed for pop
        y0 = p0.pop(0)

        if not self.mw.mol_multifit_rot_check.isChecked():
            Trot = p0.pop(0)
        if not self.mw.mol_multifit_vib_check.isChecked():
            Tvib = p0.pop(0)

        specs = []
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit == True:
                A = p0.pop(0)        
                if self.mw.mol_multifit_rot_check.isChecked():
                    Trot = p0.pop(0)
                if self.mw.mol_multifit_vib_check.isChecked():
                    Tvib = p0.pop(0)
                this_spec = A*get_mOES_spec(x, Tvib, Trot, mol_sel, self.get_instr)
                specs.append(this_spec)
        
        return np.sum(specs, axis=0) + y0


    def show_spec(self):
        self.clear_spec()
            
        min_x = 0
        max_x = 1100
        max_y = 1
        lim_unset = True
        Te = -1
        lw = -1
        
        # find wavelength range and max_y from file spec
        for plot_item in self.mw.specplot.listDataItems():
            if "file" in plot_item.name():
                x0, x1 = plot_item.dataBounds(0)
                if lim_unset:
                    min_x = x0
                    max_x = x1
                    max_y = plot_item.dataBounds(1)[1]
                    lim_unset = False
                
                min_x = min(min_x, x0)
                max_x = max(max_x, x1)

        if self.mw.mol_limit_range_check.isChecked():
            min_x = self.mw.mol_min_wl_sbox.value()
            max_x = self.mw.mol_max_wl_sbox.value()
        
        # x = np.linspace(min_x)
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked(): 
                mol_sel.load_db((min_x, max_x))
                if mol_sel.src == "mOES" and mol_sel.can_fit:
                    Trot = self.mw.mol_Trot_sbox.value()
                    Tvib = self.mw.mol_Tvib_sbox.value()
                    sim_x = np.linspace(min_x, max_x, 100000)
                    sim_y = get_mOES_spec(sim_x, Tvib, Trot, mol_sel, self.get_instr)
                    sim_y = sim_y/np.max(sim_y) * max_y

                    self.mw.plot(sim_x, sim_y, 'molecule: ' + mol_sel.label 
                                            + ' Tvib = ' + str(round(Tvib)) 
                                            + ' Trot = ' + str(round(Trot)) )
                        
                if mol_sel.src == "LIFBASE":
                    simx,simy = np.loadtxt(file_dir + "/data/mol_spec/" + mol_sel.ident + ".mod", delimiter=",").T
                    simx = simx/10 # Angstrom to nm
                    instr = self.get_instr(simx)
                    simy = fftconvolve(simy, instr/np.sum(instr), mode='same')
                    simy = simy/np.max(simy) * max_y

                    self.mw.plot(simx, simy, 'molecule: ' + mol_sel.label 
                                            + ' fixed temperature Tvib = 2500 K' 
                                            + ' Trot = 500 K' )

        self.mw.update_spec_colors()


    def fit_children(self,item):
        """ Recursivly walks through all children of selected tree item. Calls
        fit_filetree_item for each leaf. """
        if item.childCount() == 0:
            self.fit_filetree_item(item)
        else:
            for idx in range(item.childCount()):
                child = item.child(idx)
                self.fit_children(child) 
                
                
    def fit_filetree_item(self, this_item):
        """ Walks up the tree to assemble the path. """
        path = []
        current_parent = this_item.parent()
        path.append(this_item.text(0))
        while not current_parent is None:
            path.append(current_parent.text(0))
            current_item = current_parent
            current_parent = current_item.parent()
        path = os.path.join(self.mw.active_folder, *path[::-1])
        x,y = self.mw.open_file(path, this_item)   
        self.fit_spec(x+self.mw.wl_shift.value(),y,this_item.text(0))
        
    
    def fit_spec(self,x,y,label):    
        Trot0 = self.mw.mol_Trot_sbox.value()
        Tvib0 = self.mw.mol_Tvib_sbox.value()
        A0 = np.max(y)
        p0 = [0.0,]

        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit == True:
                mol_sel.load_db((x[0], x[-1]))
                y0 = A0*get_mOES_spec(x, Tvib0, Trot0, mol_sel, self.get_instr)
                A0 = A0*np.max(y)/np.max(y0)

        if not self.mw.mol_multifit_rot_check.isChecked():
            p0.append(Trot0)
        if not self.mw.mol_multifit_vib_check.isChecked():
            p0.append(Tvib0)

        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit == True:
                p0.append(A0)
                if self.mw.mol_multifit_rot_check.isChecked():
                    p0.append(Trot0)
                if self.mw.mol_multifit_vib_check.isChecked():
                    p0.append(Tvib0)

        if self.mw.mol_limit_range_check.isChecked():
            y = y[(x>self.mw.mol_min_wl_sbox.value())*(x<self.mw.mol_max_wl_sbox.value())]
            x = x[(x>self.mw.mol_min_wl_sbox.value())*(x<self.mw.mol_max_wl_sbox.value())]
        


        mol_fit_thread = QThread()
        fit_worker = MoleculeFitter(label, x, y, p0, self.molecule_selectors, 
                                    self.mw.mol_multifit_rot_check.isChecked(), 
                                    self.mw.mol_multifit_vib_check.isChecked(),
                                    self.get_instr,
                                    self.mw.mol_wl_shift_check.isChecked(),
                                    self.mw.mol_wl_stretch_check.isChecked())
        
        fit_worker.moveToThread(mol_fit_thread)
        mol_fit_thread.started.connect(fit_worker.fit)
        fit_worker.result_ready.connect(self.fit_ready)
        fit_worker.progress.connect(self.mw.update_progress_bar)
        fit_worker.finished.connect(self.mw.update_spec_colors)
        fit_worker.finished.connect(fit_worker.deleteLater)
        mol_fit_thread.finished.connect(mol_fit_thread.deleteLater)
        mol_fit_thread.start()
        # We neet to store the local objects in a "self" list to ensure
        # they are not garbage collected right after the button press
        self.mol_fit_threads.append(mol_fit_thread) 
        self.mol_fit_workers.append(fit_worker)


    def fit_ready(self, label, ans, x_fit, y_fit):
        count = self.mw.mol_fit_results_table.rowCount()
        self.mw.mol_fit_results_table.insertRow(count)

        self.mw.mol_fit_results_table.setItem(count, 0, QTableWidgetItem(label))
        header = ["file",]
        plot_label = ""
        col = 1

        if not self.mw.mol_multifit_rot_check.isChecked():
            col_count = self.mw.mol_fit_results_table.columnCount()
            if col_count < col + 1:
                self.mw.mol_fit_results_table.insertColumn(col_count)
            self.mw.mol_fit_results_table.setItem(count, col, QTableWidgetItem(str(round(ans[col],3))))
            header.append("Trot / K")
            plot_label = plot_label + " Trot=" + str(round(ans[col],0))
            col = col + 1

        if not self.mw.mol_multifit_vib_check.isChecked():
            col_count = self.mw.mol_fit_results_table.columnCount()
            if col_count < col + 1:
                self.mw.mol_fit_results_table.insertColumn(col_count)
            self.mw.mol_fit_results_table.setItem(count, col, QTableWidgetItem(str(round(ans[col],3))))
            header.append("Tvib / K")
            plot_label = plot_label + " Tvib=" + str(round(ans[col],0))
            col = col + 1


        for mol_sel in self.molecule_selectors:

            if mol_sel.isChecked() and mol_sel.can_fit == True:
                col_count = self.mw.mol_fit_results_table.columnCount()
                if col_count < col + 1:
                    self.mw.mol_fit_results_table.insertColumn(col_count)
                self.mw.mol_fit_results_table.setItem(count, col, QTableWidgetItem(str(round(ans[col],3))))
                header.append("intensity " + mol_sel.label)
                plot_label = plot_label + " " + mol_sel.label + " "
                col = col + 1

                if self.mw.mol_multifit_rot_check.isChecked():
                    col_count = self.mw.mol_fit_results_table.columnCount()
                    if col_count < col + 1:
                        self.mw.mol_fit_results_table.insertColumn(col_count)
                    self.mw.mol_fit_results_table.setItem(count, col, QTableWidgetItem(str(round(ans[col],3))))
                    header.append("Trot " + mol_sel.label)
                    plot_label = plot_label + " Trot=" + str(round(ans[col],0))
                    col = col + 1

                if self.mw.mol_multifit_vib_check.isChecked():
                    col_count = self.mw.mol_fit_results_table.columnCount()
                    if col_count < col + 1:
                        self.mw.mol_fit_results_table.insertColumn(col_count)
                    self.mw.mol_fit_results_table.setItem(count, col, QTableWidgetItem(str(round(ans[col],3))))
                    header.append("Tvib " + mol_sel.label)
                    plot_label = plot_label + " Tvib=" + str(round(ans[col],0))
                    col = col + 1
        
        self.mw.mol_fit_results_table.setColumnCount(col)
        self.mw.mol_fit_results_table.setHorizontalHeaderLabels(header)
        self.mw.mol_fit_results_table.item(count, 0).y_fit = y_fit
        self.mw.mol_fit_results_table.item(count, 0).x_fit = x_fit
        self.mw.mol_fit_results_table.item(count, 0).plot_label = plot_label

        for plot_item in self.mw.specplot.listDataItems():
            if "file" in plot_item.name() and label in plot_item.name():
                self.mw.plot(x_fit, y_fit, 'molecule: ' + plot_label)
                    
        self.mw.update_spec_colors()


    def fit(self):
        for plot_item in self.mw.specplot.listDataItems():
            if "molecule:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
                
        # self.mol_fit_results_table.setRowCount(0)

        if self.mw.mol_fit_what_combobox.currentIndex() == 0: # fit all shown
            for plot_item in self.mw.specplot.listDataItems():
                if "file" in plot_item.name():
                    x,y = plot_item.getData()
                    self.fit_spec(x,y, plot_item.name().replace('file:',''))
                    
        if self.mw.mol_fit_what_combobox.currentIndex() == 1: # fit all checked
            iterator = QTreeWidgetItemIterator(self.mw.file_list)
            while iterator.value():
                this_item = iterator.value()
                iterator += 1
                if this_item.checkState(0) == Qt.CheckState.Checked:
                    self.fit_children(this_item)    
        

    def clear_spec(self):
        for plot_item in self.mw.specplot.listDataItems():
            if "molecule:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
                    
                    # f_sim_y = interp1d(sim_x2, sim_y2, bounds_error=False, fill_value=0)
                    # sim_y = f_sim_y(x)
                
    def change_sel(self):
        num_checked = 0
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked(): 
                num_checked = num_checked + 1
        if num_checked >= 2:
            self.mw.mol_multitemp_group.show()
        else:
            self.mw.mol_multitemp_group.hide()

        # self.update_mol()
          
    
    def clear_table(self):
        self.mw.mol_fit_results_table.setRowCount(0)


    def del_table_row(self, row):
        self.mw.mol_fit_results_table.removeRow(row)


    def save_results(self):
        seperator = "\t "
        next_line = " \n"
        filename = QFileDialog.getSaveFileName(caption='Save File',
            filter='*.txt')
        table_header = ""

        if filename[0]:

            header = ("## OES toolbox result file: molecular band emission fit ## \n" +
                      "# " + str(datetime.datetime.now()) + "\n\n")
                    
            for i in range(0, self.mw.mol_fit_results_table.columnCount()):
                table_header = table_header + self.mw.mol_fit_results_table.horizontalHeaderItem(i).text() + seperator

            lines = [header, table_header + " \n"]
            for x in range(self.mw.mol_fit_results_table.rowCount()):
                this_line = ""
                for y in range(self.mw.mol_fit_results_table.columnCount()):
                    this_line = this_line + str(self.mw.mol_fit_results_table.item(x,y).text()) + seperator
                lines.append(this_line + next_line)

            try:
                f = open(filename[0], 'w', encoding="utf-8")
                f.writelines(lines)
            except:
                 mb = QMessageBox()
                 mb.setIcon(QMessageBox.Icon.Information)
                 mb.setWindowTitle('Error')
                 mb.setText('Could not save file.')
                 mb.setStandardButtons(QMessageBox.StandardButton.Ok)
                 mb.exec()


    def fit_results_rightClick(self, cursor):
        row = self.mw.mol_fit_results_table.rowAt(cursor.y())
        # col = self.mol_fit_results_table.columnAt(cursor.x())
        plot_label = self.mw.mol_fit_results_table.item(row, 0).plot_label

        menu = QMenu()
        plot_action = QAction("Plot row", checkable=True)
        del_row_action = QAction("Remove row")
        clear_action = QAction("Clear table")
        
        plotted_atm = False
        for plot_item in self.mw.specplot.listDataItems():
            if "molecule: " + plot_label in plot_item.name():
                plotted_atm = True

        if plotted_atm:
            plot_action.setChecked(True)
        else:
            plot_action.setChecked(False)

        menu.addAction(plot_action)
        menu.addAction(del_row_action)
        menu.addAction(clear_action)

        plot_action.triggered.connect(lambda: self.plotl_table_item(row, plot_action.isChecked()))
        del_row_action.triggered.connect(lambda: self.del_table_row(row))
        del_row_action.triggered.connect(lambda: self.plotl_table_item(row, False))

        clear_action.triggered.connect(self.clear_table)

        menu.exec(QtGui.QCursor.pos())


    def plotl_table_item(self, row_idx, plot):
        if plot:
            y_fit = self.mw.mol_fit_results_table.item(row_idx, 0).y_fit
            x_fit = self.mw.mol_fit_results_table.item(row_idx, 0).x_fit
            plot_label = self.mw.mol_fit_results_table.item(row_idx, 0).plot_label
            self.mw.plot(x_fit, y_fit, 'molecule: ' + plot_label)
            self.mw.update_spec_colors()
        else:
            plot_label = self.mw.mol_fit_results_table.item(row_idx, 0).plot_label
            for plot_item in self.mw.specplot.listDataItems():
                if "molecule: " + plot_label in plot_item.name():
                    self.mw.specplot.removeItem(plot_item)
            self.mw.update_spec_colors()     


    