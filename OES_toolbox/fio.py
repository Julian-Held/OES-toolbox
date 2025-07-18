import os
import datetime
from pathlib import Path
import warnings
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItemIterator, QTreeWidget,\
                                QMessageBox
from PyQt6.QtCore import Qt
from PyQt6 import sip

from sif_parser import np_open
from sif_parser.utils import extract_calibration
from . import pyAvantes
from .winspec import SpeFile
from file_read_backwards import FileReadBackwards
import csv


def guess_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(50000)).delimiter
    return delimiter


class fio():
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.active_folder = ''

    def save_plots(self):
            filename = QFileDialog.getSaveFileName(caption='Save File',
                filter='*.txt')

            if filename[0]:
                header = "## OES toolbox result file: plot export ## \n" + \
                        "# " + str(datetime.datetime.now()) + "\n\n"   
                table_header = ""

                ys = []
                xs = []
                first_x = []
                same_x = True
                max_len_x = 0
                data = []
                names = []
                for plot_item in self.mw.specplot.listDataItems():
                    # if "file" in plot_item.name():
                    x,y = plot_item.getData()
                    max_len_x = np.max([max_len_x, len(x)])
                    if len(first_x) < 1:
                        first_x = x
                    else:
                        if not np.array_equal(first_x, x):
                            same_x = False
                    
                    xs.append(x)
                    ys.append(y)
                    names.append(plot_item.name())
                
                if same_x:
                    data = np.array(xs[0])
                    data = np.column_stack([data, np.array(ys).T])
                    # ,ys)).T
                    table_header = "wavelength / nm       "
                    for name in names:
                        table_header = table_header + name + "  \t  "
                
                else:
                    data = np.zeros((2*len(names), max_len_x)) 
                    for idx, name in enumerate(names):
                        x = np.pad(xs[idx], (0, max_len_x-len(xs[idx]))) # pad to same length
                        y = np.pad(ys[idx], (0, max_len_x-len(ys[idx])))
                        table_header = table_header + name + " wavelength \t " + " data \t "
                
                        data[(2*(idx))] = x
                        data[(2*(idx))+1] = y
                    data = data.T
                
                try:
                    np.savetxt(filename[0], data, header=header+table_header)
                except:
                    mb = QMessageBox()
                    mb.setIcon(QMessageBox.Icon.Information)
                    mb.setWindowTitle('Error')
                    mb.setText('Could not save file.')
                    mb.setStandardButtons(QMessageBox.StandardButton.Ok)
                    mb.exec()


    def add_sub(self, path, tree_item):
        " Recursively walks the file structure and adds items to the file tree " 
        folder_content = sorted(os.listdir(path))
        for f in folder_content:
            item = self.mw.filetree_item(f)
            tree_item.addChild(item)
            if os.path.isdir(os.path.join(path, f)):
                self.add_sub(os.path.join(path, f), item)
    
    
    def file_list_keys(self, event):
            if event.key() == Qt.Key.Key_Delete:
                iterator = QTreeWidgetItemIterator(self.mw.file_list)
                while iterator.value():
                    this_item = iterator.value()
                    iterator += 1
                    if (self.mw.plot_combobox.currentIndex() == 0 and this_item.isSelected()):
                        sip.delete(this_item)
            else:
                QTreeWidget.keyPressEvent(self.mw.file_list, event)
            event.accept()
    
    
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(caption='Open Folder')
        if folder !="":
            if os.path.isdir(folder):
                self.mw.file_list.clear()
                self.active_folder = folder

            spec_files = list(Path(self.active_folder).iterdir())
            for f in spec_files:
                if Path(f).suffix in [".png",".jpg",".ico",".svg",".ipynb",".py",".pyc"]:
                    continue
                item = self.mw.filetree_item(f)
                item.iterdir()
                self.mw.file_list.addTopLevelItem(item)
            return folder
        

    def open_files(self):
        files = QFileDialog.getOpenFileNames(caption='Open Files')
        spec_files = files[0]
        for f in spec_files:
            item = self.mw.filetree_item(f)
            self.mw.file_list.addTopLevelItem(item)
        return ""
        

    def open_bg_file(self):
        path = QFileDialog.getOpenFileName(caption='Open background file')[0]
        self.mw.bg_extra_ledit.setText(path)                
        return path
    
    
    def load_generic_file(self, path, tree_item, content, num, file_type):
        from pandas import read_csv
        decimal = '.'
        
        import charset_normalizer #import from_bytes, from_path
        codec = charset_normalizer.from_path(path).best().encoding

        self.mw.bg_internal_check.hide()
        if file_type == "ocean_ss_txt":
            footer = 2
        else:
            with FileReadBackwards(path) as data:
                footer = 0
                while True:
                    line = data.readline()
                    if len(line.strip()) > 0:
                        if line.strip()[0].isdigit(): break
                    else:
                        footer = footer + 1
            
        sniffer = csv.Sniffer()
        try:
            with open(path, "r", encoding=codec) as data:
                pos = 0
                i = 0
                while True:
                    i = i+1
                    line = data.readline()
                    if len(line.strip()) > 0 and i > 1:
                        if line.strip()[0].isdigit(): 
                            if not '.' in line:
                                decimal = ','
                            break
                    pos = data.tell()
                delimiter = sniffer.sniff(data.read(500).replace(decimal, '.')).delimiter

                data.seek(pos) # need to take one step back to catch every line
                temp = read_csv(data, delimiter=delimiter, decimal=decimal, 
                                            skipfooter=footer, engine='python',
                                            header=None, encoding=codec)
                temp = temp.to_numpy().T
        except Exception as e:
            print(e)          

        x, y = temp[0], temp[1:]
        cols = np.shape(y)[0]
        if not content and cols > 1:
            for i in np.arange(cols):
                label = "Scan " + str(i+1)
                item = self.mw.filetree_item(label, is_content=True, num=i)
                tree_item.addChild(item)
        elif content:
            y = y[num]

        return x,y
    

    def open_avantes_txt(self, path, tree_item, content, num):
        from pandas import read_csv
        decimal = '.'
        
        import charset_normalizer #import from_bytes, from_path
        codec = charset_normalizer.from_path(path).best().encoding

        self.mw.bg_internal_check.show()
        with open(path, "r", encoding=codec) as data:
            pos = 0
            i = 0
            while True:
                i = i+1
                line = data.readline()
                if len(line.strip()) > 0 and i > 1:
                    if line.strip()[0].isdigit(): 
                        if ',' in line:
                            decimal = ','
                        break
                if line.strip().startswith("Wave"):
                    header = [e.strip() for e in line.strip().split(';') if e]
                pos = data.tell()

            data.seek(pos) # need to take one step back to catch every line
            temp = read_csv(data, delimiter=';', decimal=decimal, header=None, 
                            encoding=codec)
            temp = temp.to_numpy().T
            
        x = temp[header.index("Wave")]
        if self.mw.bg_internal_check.isChecked() == True:
            dark = temp[header.index("Dark")]
        else:
            dark = np.zeros(len(x))
        if "Sample" in header:
            y = temp[header.index("Sample")] - dark
        if "Samples" in header:
            scans = temp[header.index("Samples"):]
            if not content and len(scans) > 1:
                for i,scan in enumerate(scans):
                    label = "Scan " + str(i+1)
                    item = self.mw.filetree_item(label, is_content=True, num=i)
                    tree_item.addChild(item)
                y = scans - dark  

            else:                      
                y = scans[num] - dark
                
        if not self.mw.bg_extra_check.isChecked():
            y = y  - dark
        
        return x, y
    
    
    def open_horiba_txt(self, path, tree_item, content, num):
        "Loads Horiba txt files. Very preliminary due to limited test files."
        delimiter = "\t" # always the case?
        x1, x2 = None, None
        x = None
        scans = []
        import charset_normalizer #import from_bytes, from_path
        codec = charset_normalizer.from_path(path).best().encoding
        try:
            with open(path, "r", encoding=codec) as data:
                while True:
                    line = data.readline()
                    if not line:
                        break
                    if "Wavelength" in line:
                        if "HSen" in line:
                            line = data.readline()
                            x1 = np.array(line.strip().split(delimiter), dtype=float)
                            
                        if "HRes" in line:
                            line = data.readline()
                            x2 = np.array(line.strip().split(delimiter), dtype=float)
                    
                    if "Intensity:" in line:
                        line = data.readline()
                        while line.strip()[0].isdigit():
                            tmp = np.array(line.strip().split(delimiter), dtype=float)
                            scans.append(tmp)
                            line = data.readline()
                        break
        except Exception as e:
            print(e)  
        
        if not content and len(scans) > 1:
            for i,scan in enumerate(scans):
                label = "Scan " + str(i+1)
                item = self.mw.filetree_item(label, is_content=True, num=i)
                tree_item.addChild(item)
            y = scans

        else:                      
            y = scans[num]
            
        x = x1 # no idea why there are 2 X axis
        return x, y                
    


    def guess_file_type(self, file_head, ext, is_bin):
        if b"Data measured with spectrometer [name]:" in file_head:
            return "avantes_txt"
        if file_head[0:3] == b'AVS':
            return "avantes_raw8"
        if b"SpectraSuite" in file_head:
            return "ocean_ss_txt"
        if file_head[0:16] == b"Andor Technology":
            return "andor_sif"
        if not is_bin and ext == "asc" and b"Date and Time:" in file_head:
            return "andor_asc_r" # header top
        if not is_bin and ext == "asc":
            return "andor_asc" # header bottom or none
        if is_bin and ext == "spe":
            return "pi_spe"
        if not is_bin and b"OESi Camera Serial Number:" in file_head and \
                                            b"Test SW Version:" in file_head:
            return "horiba_txt" # fishy, tested for one file only
        if not is_bin and ext == "csv":
            return "generic_csv"        
        if not is_bin:
            return "generic_txt"          
        
    
    def open_file(self, path, tree_item, bg=True, content=False, num=0):
        """Loads a file at path. Guesses the filetype from the header.
        bg - subtract bg from seperate file?
        content - load a specific part of the file?
        num - which column to load, only if content=True
        """
        from pandas import read_csv
        decimal = '.'
        
        x,y = np.zeros(1024), np.zeros(1024)
        textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))
        file_head = open(path, 'rb').read(1024)
        is_bin = is_binary_string(file_head)
        ext = path.split('.')[-1].lower()
        file_type = self.guess_file_type(file_head, ext, is_bin)

        match file_type:
            case "avantes_txt":
                x, y = self.open_avantes_txt(path, tree_item, content, num)

            case "avantes_raw8":
                self.mw.bg_internal_check.show()
                S = pyAvantes.Raw8(path)
                x = S.getWavelength()
                if self.mw.bg_internal_check.isChecked() == True:
                    y = S.getScope()-S.getDark()
                else:
                    y = S.getScope()
                    
            case "horiba_txt":
                x,y = self.open_horiba_txt(path, tree_item, content, num)
            
            case "andor_sif":
                self.mw.bg_internal_check.hide()
                data, info = np_open(path)
                x = extract_calibration(info)
                cols = np.shape(data)[0]
                if not content and cols > 1:
                    for i in np.arange(cols):
                        label = "Scan " + str(i+1)
                        item = self.mw.filetree_item(label, is_content=True, num=i)
                        tree_item.addChild(item)
                    y = data[:][0]                        
                else:
                    y = data[num][0]
        
            case "andor_asc":
                self.mw.bg_internal_check.hide()
                delimiter = guess_delimiter(path)
                import charset_normalizer #import from_bytes, from_path
                codec = charset_normalizer.from_path(path).best().encoding
                with FileReadBackwards(path) as data:
                    pos = 0
                    while True:
                        line = data.readline()
                        if len(line.strip()) > 0:
                            if line.strip()[0].isdigit(): break

                    if '\t' in delimiter or " " in delimiter:
                        delimiter = None # for whitespace auto detect is better
                    temp = np.genfromtxt(data, delimiter=delimiter).T
                
                x, y = temp[0], temp[1:]
                cols = np.shape(y)[0]
                if not content and cols > 1:
                    for i in np.arange(cols):
                        label = "Scan " + str(i+1)
                        item = self.mw.filetree_item(label, is_content=True, num=i)
                        tree_item.addChild(item)
                elif content:
                    y = y[num]
                    
            case "pi_spe":
                self.mw.bg_internal_check.hide()
                datafile = SpeFile(path)
                data = datafile.data
                x = datafile.xaxis
                cols = np.shape(data)[0]
                if not content and cols > 1:
                    for i in np.arange(cols):
                        label = "Scan " + str(i+1)
                        item = self.mw.filetree_item(label, is_content=True, num=i)
                        tree_item.addChild(item)
                    if np.ndim(data) == 3:
                        y = data.transpose(2,0,1)[0]
                        # TODO: the 0 here gets rid of multitrack, which we should support!
                else:
                    y = data[num][:,0]
                
            case "generic_txt" | "ocean_ss_txt" | "andor_asc_r":
                x,y = self.load_generic_file(path, tree_item, content, num, file_type)

            
        # apply intensity calibration    
        if self.mw.apply_cal_check.isChecked():
            with warnings.catch_warnings(): # ignore masked element warning
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y = np.nan_to_num((y)/self.mw.cal(x), posinf=0, neginf=0)
        
        if self.mw.bg_extra_check.isChecked() and bg:
            try:
                bgx, bgy = self.open_file(self.mw.bg_extra_ledit.text(), None,
                                             bg=False, content=True, 
                                             num=self.mw.bg_extra_ledit.num)
                y = y - bgy
            except:
                print("Cannot load background")

        if len(np.shape(y)) > 1 and (np.shape(y)[0] == 1 or np.shape(y)[1] == 1):
            y = y[0] # workaround for strange behavior when loading files...
         
            
        return x, y
    
