import os
import datetime
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItemIterator, QTableWidgetItem, \
        QMessageBox, QCheckBox, QMenu
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6 import QtGui
import pyqtgraph as pg

from .Widgets import MoleculeCheckBox, SpectrumTreeItem
from .lazy_import import lazy_import
scipy = lazy_import("scipy")
Moose = lazy_import("Moose")
Moose.Simulation = lazy_import("Moose.Simulation")

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas import DataFrame

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


def make_fit_params(y, species:list[str], Trot:float, Tvib:float, sigma:float,gamma:float, mu:float, separate_Tvib=True, separate_Trot=True, vary_broadening=True, vary_shift=False) -> lmfit.parameter.Parameters:
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
        sigma (float):          Initial estimate of Gaussian width in nm.
        gamma (float):          Initial estimate of Lorentzian width in nm.
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
    params['sigma'].set(value = sigma, vary=vary_broadening, min=1e-3,max=1)
    params['gamma'].set(value=gamma, vary=vary_broadening, min=1e-3, max=1)
    params['mu'].set(value = mu, vary=vary_shift)
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

    
    def __init__(self, label, x, y, T_rot,T_vib, molecule_dbs:dict[str,"DataFrame"], sep_Trot:bool, sep_Tvib:bool, sigma:float, gamma:float, 
                                            allow_shift=False, allow_stretch=False, parent=None):
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.x, self.y = x,y
        self.T_rot = T_rot # initial value
        self.T_vib = T_vib # initial value
        self.molecule_dbs = molecule_dbs
        self.sep_Trot = sep_Trot
        self.sep_Tvib = sep_Tvib
        self.sigma = sigma
        self.gamma = gamma
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
            sigma = self.sigma,
            gamma = self.gamma,
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
        self.active_molecules: set[MoleculeCheckBox] = set()
        molecule_list_fit = [{"ident":k,"label":MOLECULE_DB_LABELS.get(k,k), "src":"Moose"} for k in MOLECULES]
        molecule_list_no_fit = [{'ident':k, "label":LIFBASE_LABELS.get(k,k), "src":"LIFBASE"} for k in LIFBASE_SIMS]
      
        for i,molecule in enumerate(molecule_list_fit):
            row,col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.mw.mol_select_grid.addWidget(this_mol_check, row, col)

        for i,molecule in enumerate(molecule_list_no_fit):
            row, col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.mw.mol_select_grid_nofit.addWidget(this_mol_check, row, col)
   
        self.mol_fit_threads = []
        self.mol_fit_workers = []

    def get_instr(self) -> tuple[float, float]:
        """Return broadening parameters for a Voigt profile (σ, γ), using the values set in the main window.
        
        Note that σ corresponds to Gaussian standard deviation, while the γ is the half-width at half-maximum (HWHM) of the Lorentzian component.

        This is compatible with the `Moose.Simulation.vgt` function, which should be used to compute the broadening profile.

        This is because the function only needs a grid-size and grid-spacing as additional input, so we don't need to know the actual wavelengths each time.

        Note: this replaces the previous approach of a shallow reference to `OES_toolbox.settings.settings.get_instr` which would return a instrumental profile.
        """
        return (self.mw.mol_instr_w.value(), self.mw.mol_instr_gamma.value())


    def get_databases(self):
        """Return a dict with all databases selected for fitting, optionally loading those that have not been loaded yet.
        
        Will filter down databases to the current active wavelength region of interest, either by active data range, or by user-specified bounds.

        When there are databases that have not yet been loaded, shows a progress dialog while loading them, to make this evident to the user.

        Some (small) datbases load fast, but others can be noticeably slow.
        """
        dbs = {}
        to_load = sum(elem.can_fit and (elem._db is None) for elem in self.active_molecules)
        if to_load>0:
            with pg.ProgressDialog("Loading databases...", cancelText=None, wait=0, busyCursor=True,maximum=to_load) as loading_dialog:
                for elem in (mol for mol in self.active_molecules if mol.can_fit):
                    if (elem._db is None):
                        loading_dialog.setLabelText(f"Loading line-by-line database: {elem.label}")
                        QtWidgets.QApplication.processEvents() # make sure the label has updated
                        elem._load_database()
                        loading_dialog+=1
        for elem in (mol for mol in self.active_molecules if mol.can_fit):
            dbs[elem.ident] = elem.get_db()
        return dbs


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
        sigma, gamma = self.get_instr()

        self.get_databases() # Cache any database; LIFBASE spectra are loaded in the loop if needed
        
        for mol_sel in self.active_molecules:
            db = mol_sel.get_db() # Loads LIFBASE if needed
            if db.shape[0]<1:
                sim_x = [min_x, max_x]
                sim_y = [0, 0]
            elif mol_sel.src == "Moose" and mol_sel.can_fit:
                sim_x = np.linspace(min_x, max_x, int((max_x - min_x) * 200))
                sim_y = Moose.model_for_fit(sim_x, sigma, gamma, 0, Trot, Tvib, A = max_y-min_y, b = min_y, sim_db = db)
            elif mol_sel.src == "LIFBASE":
                sim_x = db.wl.to_numpy()
                instr = Moose.Simulation.vgt(sigma, gamma, len(sim_x), np.abs(sim_x[1]-sim_x[0]))
                sim_y = scipy.signal.fftconvolve(db.I, instr, mode='same') # Instr is normalized
                sim_y = sim_y/np.max(sim_y) * (max_y-min_y)+min_y
            tag = ' fixed temperature' if mol_sel.src=='LIFBASE' else '' # prefix string with space if not empty
            tag_Trot = f"Trot = {Trot if mol_sel.src !='LIFBASE' else 500 :.0f} K"
            tag_Tvib = f"Tvib = {Tvib if mol_sel.src !='LIFBASE' else 2500:.0f} K"
            label = f"molecule: {mol_sel.label}{tag} {tag_Trot} {tag_Tvib}"
            self.mw.plot(sim_x, sim_y,label)

        self.mw.update_spec_colors()


    def fit_children(self,item):
        """Recursivly walks through all children of selected tree item and calls `fit_filetree_item` for each leaf.

        If the item is a leaf (i.e. has no children) itself, calls fit_filetree_item directly.
        """
        if item.childCount() == 0:
            self.fit_filetree_item(item)
        else:
            for idx in range(item.childCount()):
                child = item.child(idx)
                self.fit_children(child) 
                
                
    def fit_filetree_item(self, this_item:SpectrumTreeItem):
        """Schedule a single spectrum to be fitted.
        
        Performs `nan` masking on the input data to avoid these causing issues when fitting.
        #TODO: Arguably this should be handled in the fitting thread instead.

        Will early return when attempting to fit an item that has not been loaded.
        This is because the item can be a deeply nested item once loaded, or fail to load, instead of it being a leaf node.
        Fail early, instead of assuming the user would like to load the data and recursively fit it.
        Instead, notify through a log message and let them load the data.
        """
        if not this_item.is_loaded:
            self.mw.logger.warning(f"Cannot fit: {this_item.name()} is not loaded. Please load data first.")
            self.mw.status_msg.setText(f"Open file {this_item.name()} first, fit has been skipped.")
            # TODO: consider adding a brief popup dialog to be more loud?
            return
        self.mw.logger.info(f"Fitting: {this_item.name()}")
        x,y = this_item.spectrum
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.fit_spec(x[mask],y[mask],this_item.name())
        
    
    def fit_spec(self,x,y,label):    
        Trot0 = self.mw.mol_Trot_sbox.value()
        Tvib0 = self.mw.mol_Tvib_sbox.value()
        separate_Trot = self.mw.mol_multifit_rot_check.isChecked()
        separate_Tvib = self.mw.mol_multifit_vib_check.isChecked()

        dbs = self.get_databases()

        if self.mw.mol_limit_range_check.isChecked():
            mask = (x>self.mw.mol_min_wl_sbox.value()) & (x<self.mw.mol_max_wl_sbox.value())
            y = y[mask]
            x = x[mask]
        sigma,gamma = self.get_instr()
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
            sigma = sigma,
            gamma = gamma,
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
        # We need to store the local objects in a "self" list to ensure
        # they are not garbage collected right after the button press
        self.mol_fit_threads.append(mol_fit_thread) 
        self.mol_fit_workers.append(fit_worker)


    def fit_ready(self, label, ans:lmfit.minimizer.MinimizerResult, x_fit, y_fit):
        """Callback function that adds new results to the fit table and a plot of the fit to the plot widget.
        
        For each result, it adds a new row with the file name, and any `fraction` or `T_rot`/`T_vib` parameters.

        Will figure out of columns need to be added to the table based on the label of fit parameters.
        It wil NOT remove empty columns however!
        """
        count = self.mw.mol_fit_results_table.rowCount()
        self.mw.mol_fit_results_table.insertRow(count)
        table = self.mw.mol_fit_results_table
        current_header = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]

        self.mw.mol_fit_results_table.setItem(count, 0, QTableWidgetItem(label))
        header = [] # use lists for deterministic order, instead of sets
        plot_label = ""

        # loop through fit params and add fractions/temperatures to the table; temperatures to plot label as well
        for p in ans.params:
            if not p.startswith(("fraction","T_rot","T_vib")):
                continue
            param_label = map_param_to_label(p)
            if param_label not in current_header:
                header.append(param_label)
            if p.startswith("T"):
                plot_label += f"{param_label}={ans.params[p].value:.0f} K "

        # Figure out which columns to add to the table; use list to maintain order
        if header!=current_header:
            complete_header = current_header+[h for h in header if h not in current_header]
            table.setColumnCount(len(complete_header))
            table.setHorizontalHeaderLabels(complete_header)
        else:
            complete_header = header
        
        row_idx = table.rowCount()-1
        # reverse map of labels to parameter names for looking up the values.
        param_key_map = {map_param_to_label(k): k for k in ans.params}
        # use sets for easy intersection; order does not matter here and there won't be duplicates
        for p in list(set(param_key_map) & set(complete_header)):
            col_idx = complete_header.index(p)
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole,ans.params[param_key_map[p]].value)
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

        # this is already called through a signal when each fit worker finishes.            
        # self.mw.update_spec_colors()


    def on_fit_clicked(self):
        """Fires the fit callback when the 'Fit' button is pressed, scheduling the work.
        
        Determines which items to fit based on the selected fit mode in the combobox.

        Takes care of only fitting an item once if it both it and an ancestor are 'active'.

        Any item that has not been loaded yet will be skipped (though this actuallyhappens in `fit_filetree_item`).
        """
        # TODO: do we really want to always clear this?
        for plot_item in self.mw.specplot.listDataItems():
            if "molecule:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
                
        # When fit mode = "all shown" examine each leaf node; if "all checked" examine each checked node.
        # Note: for any item that has not been loaded, the check happens in`fit_children`/`fit_filetree_item`, no need to check here.
        must_fit_checked = self.mw.mol_fit_what_combobox.currentIndex() == 1
        flag = QTreeWidgetItemIterator.IteratorFlag.Checked if must_fit_checked else QTreeWidgetItemIterator.IteratorFlag.NoChildren
        iterator = QTreeWidgetItemIterator(self.mw.file_list, flag)
        while iterator.value():
            this_item: SpectrumTreeItem = iterator.value()
            iterator+=1
            # All-shown: Fit any childless item (see `flag``) that is marked active in the ancestry-chain
            if not must_fit_checked:
                if this_item.is_active(with_ancestors=True):
                    self.fit_filetree_item(this_item)
                continue
            # All-checked: only fit if not already fitted as child of parent
            parent:SpectrumTreeItem|None = this_item.parent()
            if parent is not None and parent._is_checked_with_ancestors():
                # If node has a parent which is checked, it will be selected for fitting as a child already
                continue
            self.fit_children(this_item)

    def clear_spec(self):
        for plot_item in self.mw.specplot.listDataItems():
            if plot_item.name().startswith("molecule:"):
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
                
    def change_sel(self):
        """Callback responsible for adding/removing `MoleculeCheckBox`s to the set of active ones (`self.active_molecules`).
        
        When active, the database or spectrum of the `MoleculeCheckBox` will be shown or fitted.

        When 2 or more line-by-line databases are active, the multi-temperature fitting group will be shown.
        """
        sender = self.mw.sender()
        if not isinstance(sender,MoleculeCheckBox):
            return
        self.active_molecules.add(sender) if sender.isChecked() else self.active_molecules.discard(sender)
        if sender.can_fit:
            self.mw.mol_multitemp_group.setVisible(sum(int(x.can_fit) for x in self.active_molecules) >= 2)
        
          
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

    
def map_param_to_label(param_name):
    """Map a parameter name to a more human-friendly label with some determinism.
    
    Attempts to match any potential species name to the MOLECULE_DB_LABELS dict.

    If any problems occur in future parsing that require adjustments, also update the test in `./tests/test_misc.py`.
    """
    parts = param_name.split("_")
    if param_name.startswith(("T_rot", "T_vib")) and len(parts) > 2:
        offset = 2
    elif param_name.startswith(("T_rot", "T_vib")):
        # edge case: "T_rot" and "T_vib" without species suffix
        return param_name.replace("_", "")
    else:
        offset = 1
    slice_pre = slice(0,offset)

    pre = "".join(parts[slice_pre])
    spec = MOLECULE_DB_LABELS.get(param_name.split("_",offset)[-1], param_name.split("_",offset)[-1])
    
    return f"{pre} {spec}".strip()
