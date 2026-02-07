import os
import datetime
import numpy as np
from PyQt6.QtWidgets import  QFileDialog, QTreeWidgetItemIterator, \
                                QTableWidgetItem, QMessageBox
from PyQt6.QtCore import Qt
from OES_toolbox.lazy_import import lazy_import
from OES_toolbox.exporters import FileExport
# from scipy import constants as const
const = lazy_import("scipy.constants")
scipy = lazy_import("scipy")


def black_body(x, T, A):
    y = (1/x**4)*(1/(np.exp(const.h*const.c/(x*1e-9*const.k*T)) - 1))
    return A*y/np.sum(y)

def black_body2(x, T, A, y0):
    y = (1/x**4)*(1/(np.exp(const.h*const.c/(x*1e-9*const.k*T)) - 1))
    return A*y/np.sum(y) + y0      

class cont_module():
    def __init__(self, mainWindow):
        self.mw = mainWindow

    def plot_continuum0(self):            
        for plot_item in self.mw.specplot.listDataItems():
            if "cont.:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
            
        for plot_item in self.mw.specplot.listDataItems():
            if "file" in plot_item.name():
                x,y = plot_item.getData()
                
        if self.mw.cont_medfilter_check.isChecked() or self.mw.cont_minfilter_check.isChecked():
            if self.mw.cont_minfilter_check.isChecked():
                from scipy.ndimage import minimum_filter1d
                y = minimum_filter1d(y, self.mw.cont_minfilter_num.value())
            if self.mw.cont_medfilter_check.isChecked():
                from scipy.signal import medfilt
                y = medfilt(y, self.mw.cont_medfilter_num.value())
            self.mw.plot(x, y, 'cont.: filtered ' + plot_item.name().replace('file:',''))
            self.mw.update_spec_colors()
            
        T0 = self.mw.cont_T0.value()
        y0 = black_body(x,T0,1)
        x_pos = np.argmax(y0)
        A0 = y[x_pos-5]/np.max(y0)

        y0 = black_body(x, T0, A0)
        self.mw.plot(x, y0, 'cont.: T = ' + str(T0))
        self.mw.update_spec_colors()
        
        
    def clear_continuum(self):        
        for plot_item in self.mw.specplot.listDataItems():
            if "cont." in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
        
    
    def clear_continuum_table(self):
        self.mw.cont_fit_results_table.setRowCount(0)


    def del_continuum_table_row(self, row):
        self.mw.cont_fit_results_table.removeRow(row)
            
        
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
        """Fit a black body spectrum to the data of an item."""
        x,y = this_item.x,this_item.y  
        self.fit_cont_spec(x,y,this_item.text(0))
        
    
    def fit_cont_spec(self,x,y,label):      
        if self.mw.cont_minfilter_check.isChecked():
            y = scipy.ndimage.minimum_filter1d(y, self.mw.cont_minfilter_num.value())
        
        if self.mw.cont_medfilter_check.isChecked():
            y = scipy.signal.medfilt(y, self.mw.cont_medfilter_num.value())
    
        T0 = self.mw.cont_T0.value()
        y0 = black_body(x,T0,1)
        x_pos = np.argmax(y0)
        A0 = y[x_pos-5]/np.max(y0)
        
        if self.mw.cont_limit_range_check.isChecked():
            y = y[(x>self.mw.cont_min_wl_box.value())*(x<self.mw.cont_max_wl_box.value())]
            x = x[(x>self.mw.cont_min_wl_box.value())*(x<self.mw.cont_max_wl_box.value())]
        
        if self.mw.cont_fit_y0_check.isChecked():
            p0 = (T0, A0, 0)
            ans, err = scipy.optimize.curve_fit(black_body2, x, y, p0=p0)
            y_fit = black_body2(x, *ans)
        else:
            p0 = (T0, A0)
            ans, err = scipy.optimize.curve_fit(black_body, x, y, p0=p0)
            y_fit = black_body(x, *ans)
            
        plot_label = f"fit T = {ans[0]:.4g} for {label}"
        self.mw.plot(x, y_fit, 'cont.: ' + plot_label)
        self.mw.update_spec_colors()
        
        count = self.mw.cont_fit_results_table.rowCount()
        self.mw.cont_fit_results_table.insertRow(count)
        self.mw.cont_fit_results_table.setItem(count, 0, QTableWidgetItem(label))
        self.mw.cont_fit_results_table.setItem(count, 1, QTableWidgetItem(str(round(ans[0],3))))
        self.mw.cont_fit_results_table.setItem(count, 2, QTableWidgetItem(str(round(ans[1],3))))
    
        self.mw.cont_fit_results_table.item(count, 0).y_fit = y_fit
        self.mw.cont_fit_results_table.item(count, 0).x_fit = x
        self.mw.cont_fit_results_table.item(count, 0).plot_label = plot_label
        
    def fit_continuum(self):
        for plot_item in self.mw.specplot.listDataItems():
            if "cont.:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
                
        self.mw.cont_fit_results_table.setRowCount(0)

        if self.mw.cont_fit_what_combobox.currentIndex() == 0: # fit all shown
            for plot_item in self.mw.specplot.listDataItems():
                if "file" in plot_item.name():
                    x,y = plot_item.getData()
                    self.fit_cont_spec(x,y, plot_item.name().replace('file:',''))
                    
        if self.mw.cont_fit_what_combobox.currentIndex() == 1: # fit all checked
            iterator = QTreeWidgetItemIterator(self.mw.file_list)
            while iterator.value():
                this_item = iterator.value()
                iterator += 1
                if this_item.checkState(0) == Qt.CheckState.Checked:
                    self.fit_children(this_item)

    def plot_cont_table_item(self, row_idx, plot):
        if plot:
            y_fit = self.mw.cont_fit_results_table.item(row_idx, 0).y_fit
            x_fit = self.mw.cont_fit_results_table.item(row_idx, 0).x_fit
            plot_label = self.mw.cont_fit_results_table.item(row_idx, 0).plot_label
            self.mw.plot(x_fit, y_fit, 'cont.: ' + plot_label)
            self.mw.update_spec_colors()
        else:
            plot_label = self.mw.cont_fit_results_table.item(row_idx, 0).plot_label
            for plot_item in self.mw.specplot.listDataItems():
                if "cont.: " + plot_label in plot_item.name():
                    self.mw.specplot.removeItem(plot_item)
            self.mw.update_spec_colors()  
