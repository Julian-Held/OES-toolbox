import os
import datetime
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from astropy.table import Table as aTable

file_dir = os.path.dirname(os.path.abspath(__file__))

import sys
import os.path
home = os.path.expanduser('~')
sys.path.append(home + '/.local/lib/')
import owlspec as owl # type: ignore


class NISTloader(QObject):
    def __init__(self, spec, wl_range, max_y, Te=-1, lw=-1, parent=None):
        super(self.__class__, self).__init__(parent)
        self.spec = spec
        self.wl_range = wl_range
        self.max_y = max_y
        self.Te = Te
        self.lw = lw
    
    finished = pyqtSignal()
    result_ready = pyqtSignal(np.ndarray, np.ndarray, str)
    data_ready = pyqtSignal(str, aTable)
    progress = pyqtSignal(int)
    
    def run(self):
        self.progress.emit(1)
        try:
            ele_spec = owl.spectrum(self.spec, wl_range=self.wl_range)
            nist_data = ele_spec.get_linedata()
            self.data_ready.emit(self.spec, nist_data)
            if self.Te == -1 and self.lw == -1:
                x,y = ele_spec.table_to_ident(nist_data)
            if self.Te > 0 and self.lw == -1:
                x,y = ele_spec.table_to_ident_LTE(nist_data, self.Te)
            y = y/np.max(y)
        except Exception as e:
            print(e)
            x = np.linspace(self.wl_range[0],self.wl_range[-1],10)
            y = np.zeros(len(x))
            
        self.result_ready.emit(x,self.max_y*y, "NIST: "+self.spec)
        self.progress.emit(-1)
        self.finished.emit()


class ident_module():
    def __init__(self, mainWindow):
        self.mw = mainWindow

        self.nist_threads = []
        self.nist_workers = []
        
    def update_spec_ident(self): 
        """ Loads NIST spectra and plots them. """
        min_x = 0
        max_x = 1100
        max_y = 1
        lim_unset = True
        Te = -1
        
        # remove ident specs
        for plot_item in self.mw.specplot.listDataItems():
            if "NIST" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
                
        # clear ident table
        self.mw.ident_table.setRowCount(0)
        
        # find wavelength range from file spec
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
        
        # fetch options
        if self.mw.ident_int_cbox.currentIndex() == 1:
            Te = self.mw.ident_Te.value()
        
        # spectra from the identify module 
        spec_string = self.mw.spec_line.text()
        if len(spec_string)>1:
            for spec in spec_string.split(','):
                if '-' in spec:
                    spec1 = spec.split('-')[0]
                    element, charge1 = owl.util.parse_spectroscopic_name(spec1)
                    spec2 = spec.split('-')[1]
                    spec2 = element + ' ' + spec2
                    _, charge2 = owl.util.parse_spectroscopic_name(spec2)
                    subspecs = []
                    for charge in np.arange(charge1,charge2+1):
                        this_spec = owl.util.get_spectroscopic_name(element, charge)
                        subspecs.append(this_spec)
                else:
                    subspecs = [spec]
                    
                for spec in subspecs:
                    nist_thread = QThread()
                    nist_worker = NISTloader(spec, (min_x,max_x), max_y, Te=Te)
                    nist_worker.moveToThread(nist_thread)
                    nist_thread.started.connect(nist_worker.run)
                    nist_worker.data_ready.connect(self.table_add)
                    nist_worker.result_ready.connect(self.mw.plot)
                    nist_worker.progress.connect(self.mw.update_progress_bar)
                    nist_worker.finished.connect(self.mw.update_spec_colors)
                    nist_worker.finished.connect(nist_worker.deleteLater)
                    nist_thread.finished.connect(nist_thread.deleteLater)
                    nist_thread.start()
                    # We neet to store the local objects in a "self" list to ensure
                    # they are not garbage collected right after the button press
                    self.nist_threads.append(nist_thread) 
                    self.nist_workers.append(nist_worker)

        self.mw.update_spec_colors()
    
    
    def clear_spec_ident(self):
        # for thread in self.nist_threads:
        #     thread.terminate()
        # TODO: figure out how to stop process 
        
        # clear ident table
        self.mw.ident_table.setRowCount(0)
        
        for plot_item in self.mw.specplot.listDataItems():
            if "NIST" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
    
    
    def table_add(self, spec, nist_data):
        " Add nist_data to the ident table and sort for wl "
        for i,line in enumerate(nist_data):
            c1 = str(line['Observed'])
            # astroquery does not filter out headings in the middle of the table
            if 'Observed' in c1 or "Wavelength" in c1 or "nm" in c1:
                continue
            
            if not np.ma.is_masked(line['Observed']):
                count = self.mw.ident_table.rowCount()
                self.mw.ident_table.insertRow(count)
                self.mw.ident_table.setItem(count, 0, QTableWidgetItem(str(spec)))
                wl_col = QTableWidgetItem()
                wl_col.setData(0, round(float(line['Observed']),3))
                self.mw.ident_table.setItem(count, 1, wl_col)
                self.mw.ident_table.setItem(count, 2, QTableWidgetItem(str(line['Rel.'])))
                self.mw.ident_table.setItem(count, 3, QTableWidgetItem(str(line['Aki'])))
                self.mw.ident_table.setItem(count, 4, QTableWidgetItem(str(line['Ei           Ek'])))
                self.mw.ident_table.setItem(count, 5, QTableWidgetItem(str(line['Lower level'])))
                self.mw.ident_table.setItem(count, 6, QTableWidgetItem(str(line['Upper level'])))

        self.mw.ident_table.sortItems(1, Qt.SortOrder.AscendingOrder)


    def ident_int_changed(self, index_selected):
        if index_selected == 0: # rel int
            self.mw.ident_Te.hide()
            self.mw.ident_Te_label.hide()
        else: # LTE -> show Te input box
            self.mw.ident_Te.show()
            self.mw.ident_Te_label.show()
            
            
    def save_NIST_data(self):
        seperator = "\t "
        next_line = " \n"
        filename = QFileDialog.getSaveFileName(caption='Save File', filter='*.txt')

        if filename[0]:

            header = ("## OES toolbox result file: NIST data ## \n" +
                        "# " + str(datetime.datetime.now()) + "\n\n")
                    
            table_header = ("Ion" + seperator + "wl / nm" +  seperator + 
                "rel. intensity" + seperator + "Aik" + seperator + "Ek - Ei" + 
                seperator + "lower configuration" + seperator + 
                "upper configuration" + seperator +  "\n")

            lines = [header, table_header]
            for x in range(self.mw.ident_table.rowCount()):
                this_line = ""
                for y in range(self.mw.ident_table.columnCount()):
                    this_line = this_line + str(self.mw.ident_table.item(x,y).text()) + seperator
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