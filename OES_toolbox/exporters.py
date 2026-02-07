"""Module for handling exporting data to files, either data files or images.

Since exporting can rely on user interaction in the GUI, this functionality is kept separate from `file_handling.py`

This makes the file reading logic useable without Qt being installed.
"""
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidget, QInputDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from OES_toolbox.lazy_import import lazy_import
from OES_toolbox._version import version
pd = lazy_import("pandas")

TOOLBOXSTYLE = "OES_toolbox.ui.jh-paper"

class FileExport:
    """Class with methods for exporting data and/or plots to various file types."""
    pickedLast=None
    lastFolder = None

    @classmethod
    def get_save_path(cls)-> None|Path:
        """Get a file path for saving a file."""
        filename,_ = QFileDialog.getSaveFileName(caption='Save File', 
                                                 filter="""Text file (tab-separated)(*.txt);;
                                                 Text file (comma-separated)(*.csv);;
                                                 Microsoft Excel (*.xlsx);;Apache Parquet (*.par)
                                                 """, directory=cls.lastFolder)
        if filename=="":
            return
        filename = Path(filename)
        cls.lastFolder = filename.parent.as_posix()
        return filename 

    @staticmethod
    def add_attrs(df:pd.DataFrame, kind:str):
        df.attrs =  {
            "OES toolbox version": version,
            "Result file": kind,
            "Exported on": datetime.datetime.now().isoformat(sep=" ",timespec='seconds'),
        }
    
    @classmethod
    def store_dataframe(cls,path:Path,data:pd.DataFrame):
        try:
            if path.suffix.lower()==".par":
                data.to_parquet(path)
                return
            elif "xls" in path.suffix.lower():
                    
                    with pd.ExcelWriter(path, mode="w", engine='xlsxwriter') as writer:
                        if isinstance(data.columns, pd.MultiIndex):
                            # write header first to avoid empty row bug after header in pandas for a MultiIndex
                            data.drop(data.index).to_excel(writer,index_label="Index") 
                            data.to_excel(writer, startrow=data.columns.nlevels-1, header=False)
                        else:
                            data.to_excel(writer,header=True, index=False)
                        sheet = writer.book.add_worksheet("Export info")
                        header_format = writer.book.add_format({"bold": True})
                        sheet.write(0, 0, "OES toolbox version:", header_format)
                        sheet.write(0, 1, version)
                        sheet.write(1, 0, "Exported on:", header_format)
                        sheet.write(1, 1, data.attrs["Exported on"])
                        sheet.write(2, 0, "Export:", header_format)
                        sheet.write(2, 1, data.attrs["Result file"])
                        sheet.set_column_pixels(0, 0, 170)
                    return
            header = (
                f"## OES toolbox ({version}) result file: {data.attrs['Result file']} \n"
                f"# Exported on {data.attrs['Exported on']}\n"
            )
            if isinstance(data.columns, pd.MultiIndex):
                # use a space (`\s`) instead of empty string for text export to avoid `Unnamed columns`
                cols = data.columns.to_frame()
                cols = cols[cols.columns.drop("region")] # not used?!
                for idx, col in enumerate(cols.columns):                        
                    cols[col] = cols[col].replace(""," ")
                    cols.iloc[(0, idx)] = "# " + cols.iloc[(0, idx)] # comment symbol to header lines

                data.columns = pd.MultiIndex.from_frame(cols)

            if path.suffix.lower()==".csv":
                path.write_text(header,encoding='utf-8')
                data.to_csv(path,sep=",", decimal=".", encoding="utf-8", mode="a", index=False)
            
            else: # e.g. .txt
                path.write_text(header,encoding='utf-8')
                data.to_csv(path,sep='\t', decimal=".", encoding="utf-8", mode="a", index=False)

        except Exception as e:
            QMessageBox.warning(
                None,
                "Error saving file",
                f"Cannot save file {path.as_posix()}\n\nError:\n{e}",
                QMessageBox.StandardButton.Ok
            )
    
    @classmethod
    def graph_to_matplotlib(cls, graph) -> Figure:
        plt.style.use('default') # Make sure to clear/reset styling to default for predictable results.
        plt.style.use(TOOLBOXSTYLE)
        xlim,ylim = graph.getViewBox().viewRange()
        fig = plt.figure(dpi=300)
        for plot_item in graph.listDataItems():
            pen = plot_item.opts['pen']            
            style_kws = {
                "c":pen.color().name(), 
                "zorder":plot_item.zValue(),
                "label": plot_item.name(),
                "lw": pen.width(),
            }
            match pen.style().name.lower():
                case "solidline":
                    style_kws['ls'] = "-"
                case "dotline":
                    style_kws['ls'] = ":"
                case 'dashline':
                    style_kws['ls'] = "--"
                case "nopen":
                    style_kws['ls']=''
            if plot_item.name().startswith("file"):
                style_kws["label"]=plot_item.name().strip("file:").strip()
            if plot_item.name().startswith("cont"):
                style_kws["lw"] = 1
            elif plot_item.name().startswith('NIST'):
                style_kws["lw"] = 0.5
            x,y = plot_item.getData()
            plt.plot(x,y,**style_kws)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if graph.plotItem.legend.isVisible() and (len(graph.plotItem.legend.items)>0):
            plt.legend()
        plt.xlabel(graph.getAxis("bottom").label.toPlainText())
        plt.ylabel(graph.getAxis("left").label.toPlainText())
        return fig
    
    @staticmethod
    def matplotlib_to_image(fig) -> QImage:
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = QImage(
            canvas.buffer_rgba(), 
            int(fig.figbbox.width),
            int(fig.figbbox.height),
            QImage.Format.Format_RGBA8888_Premultiplied
        )
        return img
    
    @classmethod
    def save_plot_data(cls, plot,kind:str="plot export"):
        filename = cls.get_save_path()
        if filename is None:
            return
        xlabel = plot.getAxis("bottom").label.toPlainText().strip()
        ylabel = plot.getAxis("left").label.toPlainText().strip()
        data = []
        for plot_item in plot.listDataItems():
            x,y = plot_item.getData()
            part_names = [part.strip() for part in plot_item.name().split(": ")]
            match len(part_names):
                case 2:
                    part_names.extend([""] * 2)
                case 3:
                    part_names.insert(2, "")
            data.append(pd.DataFrame({(*part_names,label):values for label, values in zip([xlabel,ylabel],[x,y])}))
        
        df = pd.concat(data,axis=1)
        df.columns.set_names(("type","path","region","label","axis"), inplace=True)
        cls.add_attrs(df, kind)
        cls.store_dataframe(filename,df)


    @classmethod
    def save_table(cls,table:"QTableWidget"):
        action_name = table.sender().text()
        filename = cls.get_save_path()
        if filename is None:
            return
        if "NIST" in action_name:
            kind="NIST data"
        elif "Molecule" in action_name:
            kind = "molecular band emission fit"
        elif "Continuum" in action_name:
            kind = "continuum emission"
        else:
            raise ValueError(f"Unknown action name for `save_table`: {action_name=}")
        
        data = {table.horizontalHeaderItem(c).text(): [table.item(r,c).text().strip() if table.item(r,c) is not None else "" for r in range(table.rowCount())] for c in range(table.columnCount())}
        df = pd.DataFrame(data)
        df = df.convert_dtypes()
        cls.add_attrs(df, kind=kind)
        cls.store_dataframe(filename, df)


