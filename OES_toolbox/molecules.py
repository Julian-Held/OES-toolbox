import os
import datetime
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItemIterator, QTableWidgetItem, \
        QMessageBox, QCheckBox, QMenu
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6 import QtGui
import pyqtgraph as pg

from .Widgets import MoleculeCheckBox
from .lazy_import import lazy_import
scipy = lazy_import("scipy")
Moose = lazy_import("Moose")

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
    from collections.abc import Callable

import lmfit
from Moose.lmfit import multi_species_objective

PARAMARGS = ('vary',"min",'max')

DEFAULT_PARAMS = Moose.default_params
DEFAULT_PARAMS["T_vib"]['max'] = 25000
DEFAULT_PARAMS["T_rot"]['max'] = 25000

# Exclude high J Swan band database; can quickly run OOM without care
MOLECULES = [f for f in Moose.database_files if "J300" not in f]
MOLECULE_DB_LABELS = {
    "C2_swan":"C₂ Swan",
    "CNBX": "CN (B-X)",
    "N2CB": "N₂ (C-B)",
    "N2PlusBX": "N₂⁺ (B-X)",
    "NHAX": "NH (A-X)",
    "NOBX": "NO (B-X)",
    "OHAX": "OH (A-X)"
}
"""Mapping of database names to better looking label for use in UI."""

LIFBASE_SIMS = [f.stem for f in Path(f"{__file__}").parent.joinpath("data/mol_spec").glob("*.mod")]
LIFBASE_LABELS = {
    'NOAX': "NO (A-X)",
    'NOCX': "NO (C-X)",
    'NODX': "NO (D-X)",
    'CHAX': "CH (A-X)",
    'CHBX': "CH (B-X)",
    'CHCX': "CH (C-X)",
    'SiHAX': "SiH (A-X)",
    'CFAX': "CF (A-X)",
    'CFBX': "CF (B-X)",
    'CFCX': "CF (C-X)",
    'CFDX': "CF (D-X)",
    'CNBX': "CN (B-X)"
}

# constants for UserRoles for storing fit results with QTableWidgetItems
PlotItemRole = Qt.ItemDataRole.UserRole+1
FitResultRole = Qt.ItemDataRole.UserRole+2


def make_fit_params(y, species:list[str], Trot:float, Tvib:float, mu:float, separate_Tvib=True, separate_Trot=True, vary_broadening=True, vary_shift=False) -> lmfit.parameter.Parameters:
    """Construct suitable fit parameters with bounds for fitting a spectrum with the `Moose.lmfit.multi_species_objective` function.

    For each element in the `species` list, it will add `fraction` and optional `T_rot`/`T_vib` parameters, as appropriate.
    
    Will calculate bounds and set parameters as free/fixed depending on provided arguments, which can come from the UI state.

    The parameter bounds are made assuming that `multi_species_objective` will be called with the `normalize` kwargs set to `True`.

    This means that `fraction` should be interpreted as a non-normalized `weight` to the total intensity, and is affected by strength of emitter.

    (Stronger emitters will cause a lower weight).

    Arguments:
        y (NDArray):            The array of y-data, to calculate offset parameter `b` and total amplitude `A` from.
        species (list[str]):    List of species names, for the species that will be used in the model objective function
        Trot (float):           Initial estimate of T_rot
        Tvib (float):           Initial estimate of T_vib
        mu (float):             Initial wavelength shift in nm.
        separate_Tvib (bool):   Flag to use different vibrational temperatures for each species
        separate_Trot (bool):   Flag to use different rotational temperatures for each species
        vary_broadening (bool): Flag to optimize broadening parameters during fit, or leave them fixed.
        vary_shift (bool):      Flag to vary the wavelength shift during fitting, or leave it fixed.
    
    Returns:
        Parameters:     A lmfit.Parameters instance with parameter values and bounds configured according to the current UI settings.
    """
    #TODO: decide if using `normalize=True` or `False`.
    separate_Tvib = separate_Tvib and (len(species)>1)
    separate_Trot = separate_Trot and (len(species)>1)
    y_min = y.min()
    y_max = y.max()
    y_diff = y_max-y_min
    # var = (y-y_min).std()
    params = lmfit.create_params(**DEFAULT_PARAMS)
    params.add("b", y_min, True, y_min - y_diff, y_min + y_diff)
    params.add("A", y_diff, vary =True, min = 0, max = y_diff*1.5)
    # params.pop("A")  # use this if using `normalize=False`
    params['sigma'].vary = vary_broadening
    params['gamma'].vary = vary_broadening
    params['mu'].vary = vary_shift
    params['mu'].value = mu
    params['T_rot'].value = Trot
    params['T_vib'].value = Tvib
    weight = 1/len(species)*y_diff
    if separate_Tvib:
        params.pop("T_vib")
    if separate_Trot:
        params.pop("T_rot")
    for specie in species:
        if separate_Trot:
            params.add(f"T_rot_{specie}", value = Trot, **{k:v for k,v in DEFAULT_PARAMS['T_vib'].items() if k in PARAMARGS})
        if separate_Tvib:
            params.add(f"T_vib_{specie}", value = Tvib, **{k:v for k,v in DEFAULT_PARAMS['T_vib'].items() if k in PARAMARGS})
        params.add(f"fraction_{specie}", weight,vary=True,min=0,max=1) # adjust if using `normalize=False`
    return params


class MoleculeFitter(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(str, lmfit.minimizer.MinimizerResult, np.ndarray, np.ndarray)
    progress = pyqtSignal(int)

    
    def __init__(self, label, x, y, T_rot,T_vib, molecule_dbs:dict[str,"DataFrame"], sep_Trot:bool, sep_Tvib:bool, instr_func, 
                                            allow_shift=False, allow_stretch=False, parent=None):
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.x, self.y = x,y
        self.T_rot = T_rot # initial value
        self.T_vib = T_vib # initial value
        self.molecule_dbs = molecule_dbs
        self.sep_Trot = sep_Trot
        self.sep_Tvib = sep_Tvib
        self.get_instr = instr_func # function copy from Window class
        self.stop = False # stop flag invoked by button press
        self.allow_shift = allow_shift # Plot data is shifted itself, no need for a shift value if using plot data.
        self.allow_stretch =  allow_stretch

    def fit(self):
        self.progress.emit(1)
        params=make_fit_params(
            self.y,
            species=self.molecule_dbs,
            Trot = self.T_rot,
            Tvib=self.T_vib,
            mu = 0, # Plot data is shifted already
            vary_shift = self.allow_shift,
            separate_Tvib=self.sep_Tvib, 
            separate_Trot=self.sep_Trot,
            vary_broadening=True
        )
        # TODO: investigate if error handling is needed.
        result = lmfit.minimize(
            multi_species_objective,
            params,
            args=(self.x,),
            kws=
            {
                "y":self.y,
                "normalize":True,
                **self.molecule_dbs
            },
            ftol=1e-10,
            max_nfev = 2000
        )
        y_fit = multi_species_objective(result.params, x = self.x, normalize = True, **self.molecule_dbs)
        self.result_ready.emit(self.label, result, self.x, y_fit)
        self.progress.emit(-1)
        self.finished.emit()



##############################################################################
# <------------------------- molecules module -----------------------------> #
##############################################################################   
class molecule_module:
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.get_instr = self.mw.settings.get_instr 
        molecule_list_fit = [{"ident":k,"label":MOLECULE_DB_LABELS.get(k,k), "src":"mOES"} for k in MOLECULES]
        molecule_list_no_fit = [{'ident':k, "label":LIFBASE_LABELS.get(k,k), "src":"LIFBASE"} for k in LIFBASE_SIMS]
        
        self.molecule_selectors:list[MoleculeCheckBox] = []
      
        for i,molecule in enumerate(molecule_list_fit):
            row,col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.molecule_selectors.append(this_mol_check)
            self.mw.mol_select_grid.addWidget(this_mol_check, row, col)

        for i,molecule in enumerate(molecule_list_no_fit):
            row, col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.molecule_selectors.append(this_mol_check)
            self.mw.mol_select_grid_nofit.addWidget(this_mol_check, row, col)
   
        self.mol_fit_threads = []
        self.mol_fit_workers = []

    def show_spec(self):
        self.clear_spec()
            
        min_x = 0
        max_x = 1100
        max_y = 1
        lim_unset = True
        Te = -1
        lw = -1
        
        
        min_x, max_x, min_y ,max_y = self.mw.get_bounds()
        min_y = max(min_y, 0) # clamp to minimum of 0 for visualization

        Trot = self.mw.mol_Trot_sbox.value()
        Tvib = self.mw.mol_Tvib_sbox.value()
        
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked(): 
                db = mol_sel.get_db((min_x, max_x)) # will cache if not loaded yet
                if db.shape[0]<1:
                    sim_x = [min_x, max_x]
                    sim_y = [0, 0]
                elif mol_sel.src == "mOES" and mol_sel.can_fit:
                    sim_x = np.linspace(min_x, max_x, int((max_x - min_x) * 200))
                    #TODO: support broadening once more
                    sim_y = Moose.model_for_fit(sim_x, 0.001, 0.001, 0, Trot, Tvib, A = max_y-min_y, b = min_y, sim_db = db)
                elif mol_sel.src == "LIFBASE":
                    instr = self.get_instr(db.wl)
                    sim_x = db.wl
                    sim_y = scipy.signal.fftconvolve(db.I, instr / np.sum(instr), mode='same')
                    sim_y = sim_y/np.max(sim_y) * (max_y-min_y)+min_y
                tag = ' fixed temperature' if mol_sel.src=='LIFBASE' else '' # prefix string with space if not empty
                tag_Trot = f"Trot = {Trot if mol_sel.src !='LIFBASE' else 500 :.0f} K"
                tag_Tvib = f"Tvib = {Tvib if mol_sel.src !='LIFBASE' else 2500:.0f} K"
                label = f"molecule: {mol_sel.label}{tag} {tag_Trot} {tag_Tvib}"
                self.mw.plot(sim_x, sim_y,label)

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
        self.mw.logger.info(f"Fitting: {this_item.name()}")
        x,y = this_item.spectrum
        self.fit_spec(x,y,this_item.name())
        
    
    def fit_spec(self,x,y,label):    
        Trot0 = self.mw.mol_Trot_sbox.value()
        Tvib0 = self.mw.mol_Tvib_sbox.value()
        separate_Trot = self.mw.mol_multifit_rot_check.isChecked()
        separate_Tvib = self.mw.mol_multifit_vib_check.isChecked()

        dbs = {}
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit is True:
                db = mol_sel.get_db()
                if db.shape[0]>0:
                    dbs[mol_sel.ident] = db

        if self.mw.mol_limit_range_check.isChecked():
            mask = (x>self.mw.mol_min_wl_sbox.value()) & (x<self.mw.mol_max_wl_sbox.value())
            y = y[mask]
            x = x[mask]

        mol_fit_thread = QThread()
        fit_worker = MoleculeFitter(
            label = label, 
            x = x, 
            y = y, 
            T_rot = Trot0,
            T_vib = Tvib0, 
            molecule_dbs = dbs,
            sep_Trot = separate_Trot, 
            sep_Tvib = separate_Tvib,
            instr_func = self.get_instr,
            allow_shift = self.mw.mol_wl_shift_check.isChecked(),
            allow_stretch = self.mw.mol_wl_stretch_check.isChecked()
        )
        
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


    def fit_ready(self, label, ans:lmfit.minimizer.MinimizerResult, x_fit, y_fit):
        def get_species_name(param_name,split_count=2):
            """Construct a name for a species from a parameter name, if applicable."""
            parts = param_name.split("_", split_count)
            return MOLECULE_DB_LABELS.get(parts[split_count],parts[split_count].replace("_"," ")) if len(parts)>split_count else ""

        count = self.mw.mol_fit_results_table.rowCount()
        self.mw.mol_fit_results_table.insertRow(count)
        table = self.mw.mol_fit_results_table
        current_header = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]

        self.mw.mol_fit_results_table.setItem(count, 0, QTableWidgetItem(label))
        header = ["file",]
        plot_label = ""
        #TODO: figure out using clean labels for the table, while mapping them to parameter names to populate cells
        print("========================")
        print("FIT result:")

        for p in ans.params:
            print(f"{p}: {ans.params[p]}")
            match p:
                case s if "fraction" in s:
                    # s = s.replace("_", " ")
                    species_name = get_species_name(s,1)
                    # col_name = f"fraction {species_name}"
                    # if col_name not in current_header:
                    #     header.append(col_name)
                    if s not in current_header:
                        header.append(s)
                case s if "T_rot" in s:
                    # s = s.replace("rot_","rot ")+" / K"
                    species_name = get_species_name(s,2)
                    # col_name = f"Trot {species_name} / K"
                    # if col_name not in current_header:
                    #     header.append(col_name)
                    if s not in current_header:
                        header.append(s)                   
                    plot_label += f"Trot {species_name}={ans.params[p].value:.0f} K "
                case s if "T_vib" in s:
                    # s = s.replace("vib_","vib ")+" / K"
                    species_name = get_species_name(s,2)
                    # col_name = f"Tvib {species_name} / K"
                    # if col_name not in current_header:
                    #     header.append(col_name) 
                    if s not in current_header:
                        header.append(s)                   
                    plot_label += f"Tvib {species_name}={ans.params[p].value:.0f} K "
                case _:
                    continue
        print("========================")
        print(header,current_header)
        if header!=current_header:
            complete_header = current_header+[h for h in header if h not in current_header]
            table.setColumnCount(len(complete_header))
            table.setHorizontalHeaderLabels(complete_header)
        else:
            complete_header = header
        
        row_idx = table.rowCount()-1
        for p in list(set(ans.params)&set(complete_header)):
            col_idx = complete_header.index(p)
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole,ans.params[p].value)
            table.setItem(row_idx,col_idx,item)

        self.mw.mol_fit_results_table.item(count, 0).y_fit = y_fit
        self.mw.mol_fit_results_table.item(count, 0).x_fit = x_fit

        # Create the plot item and associate it with the 'file name' cell as UserData, same as fit result, using `PlotItemRole`.
        # Untill we use a proper MVC pattern, this may be a good start point to show e.g. residual etc.
        plot_item = pg.PlotDataItem(x=x_fit,y=y_fit, name = f"molecule: {plot_label.strip()}")
        table.item(row_idx, 0).setData(PlotItemRole,plot_item)
        table.item(row_idx, 0).plot_item = plot_item
        table.item(row_idx, 0).setData(FitResultRole,ans)
        # TODO: consider how to avoid this loop, perhaps we need a reference to the object to be passed along?
        is_shown = f"file: {label.strip()}" in {item.name() for item in self.mw.specplot.listDataItems()}
        if is_shown:
            self.mw.specplot.addItem(plot_item, ignoreBounds=True)

                    
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
                    # Remove any element that has either x or y nan
                    mask = ~np.isnan(x)& ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    self.fit_spec(x,y, plot_item.name().replace('file:',''))
                    
        if self.mw.mol_fit_what_combobox.currentIndex() == 1: # fit all checked
            iterator = QTreeWidgetItemIterator(self.mw.file_list,QTreeWidgetItemIterator.IteratorFlag.Checked)
            while iterator.value():
                this_item = iterator.value()
                iterator += 1
                # Fit only when not already fitted as child of parent
                if this_item.parent() is not None:
                    if not this_item.parent()._is_checked_with_ancestors():
                        self.fit_children(this_item)
                else:
                    self.fit_children(this_item) 

    def clear_spec(self):
        for plot_item in self.mw.specplot.listDataItems():
            if plot_item.name().startswith("molecule:"):
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
                
    def change_sel(self):
        num_checked = 0
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit: 
                num_checked = num_checked + 1
        self.mw.mol_multitemp_group.setVisible(num_checked>=2)
          
    def clear_table(self):
        """Delete table items row by row to ensure proper cleanup of associated plot items.
        
        Must traverse the table in reverse order to avoid index errors as rowCount changes during iteration.
        """
        for i in range(self.mw.mol_fit_results_table.rowCount()-1,-1,-1):
            self.del_table_row(i)

    def del_table_row(self, row):
        """Delete a row from the molecure fit results table and remove the associated plot item."""
        plot_item = self.mw.mol_fit_results_table.item(row, 0).data(PlotItemRole)
        self.mw.specplot.removeItem(plot_item)
        plot_item.deleteLater()
        self.mw.mol_fit_results_table.removeRow(row)

    def del_table_col(self,col):
        """Delete a column from the fit result table.
        
        Will not remove the first column (i.e. the plot item label), since elements in this column contain required extra data in the UserRoles.
        """
        if col==0:
            return
        self.mw.mol_fit_results_table.removeColumn(col)

    def fit_results_rightClick(self, cursor):
        """Show a right-click menu to interact with the fit result table."""
        row = self.mw.mol_fit_results_table.rowAt(cursor.y())
        col = self.mw.mol_fit_results_table.columnAt(cursor.x())
        item = self.mw.mol_fit_results_table.itemAt(cursor)
        # first column (i.e. the plot label) contains the extra UserRoles (FitResultRole, PlotItemRole)
        item_col0 = self.mw.mol_fit_results_table.item(row, 0)
        if item is None:
            plotted_atm = False
        else:
            plot_item = item_col0.data(PlotItemRole)
            plotted_atm = plot_item in set(self.mw.specplot.listDataItems())

        menu = QMenu()
        plot_action = QAction("Plot row", checkable=True)
        del_row_action = QAction("Remove row")
        del_col_action = QAction("Remove column")
        clear_action = QAction("Clear table")

        if plotted_atm:
            plot_action.setChecked(True)
        else:
            plot_action.setChecked(False)
        if item:
            menu.addAction(plot_action)
            plot_action.triggered.connect(lambda: self.plotl_table_item(row, not plotted_atm))
            menu.addAction(del_row_action)
            del_row_action.triggered.connect(lambda: self.del_table_row(row))
        if col !=0:
            menu.addAction(del_col_action)
        menu.addAction(clear_action)

        del_col_action.triggered.connect(lambda: self.del_table_col(col))

        clear_action.triggered.connect(self.clear_table)

        menu.exec(QtGui.QCursor.pos())


    def plotl_table_item(self, row_idx, plot:bool):
        """Add or remove a fit result plot to the graph, depending on if it is currently drawn or not."""
        plot_item = self.mw.mol_fit_results_table.item(row_idx, 0).data(PlotItemRole)
        if plot:
            self.mw.specplot.addItem(plot_item) 
        else:
            self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()


    