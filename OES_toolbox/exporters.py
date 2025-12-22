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
        filename,_ = QFileDialog.getSaveFileName(caption='Save File', filter="Apache Parquet (*.par);;Text file (comma-separated)(*.txt *.csv)", directory=cls.lastFolder)
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
            header = (
                f"## OES toolbox ({version}) result file: {data.attrs['Result file']} ##\n"
                f"#  {data.attrs['Exported on']}\n\n"
            )
            path.write_text(header,encoding='utf-8')
            data.to_csv(path,sep=",", decimal=".",encoding="utf-8",mode="a",index=False)
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
        ys = []
        xs = []
        names = []
        for plot_item in plot.listDataItems():
            x,y = plot_item.getData()
            xs.append(x)
            ys.append(y)
            names.append(plot_item.name())
        
        names = np.concatenate([np.repeat(name,x.shape[0]) for name,x in zip(names,xs, strict=True)])
        df=pd.DataFrame(
            {
                "name":names,
                plot.getAxis("bottom").label.toPlainText().strip(): np.concatenate(xs),
                plot.getAxis("left").label.toPlainText().strip():np.concatenate(ys)
            }
        )
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
        
        data = {table.horizontalHeaderItem(c).text(): [table.item(r,c).text().strip() for r in range(table.rowCount())] for c in range(table.columnCount())}
        df = pd.DataFrame(data)
        df = df.astype({col:float if col not in ['file',"Ion","conf. lower","conf. upper"] else str for col in df.columns })
        cls.add_attrs(df, kind=kind)
        cls.store_dataframe(filename, df)


