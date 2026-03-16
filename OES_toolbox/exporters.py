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

from pyqtgraph import GraphicsScene
from pyqtgraph.exporters.Matplotlib import MatplotlibExporter, MatplotlibWindow, _symbol_pg_to_mpl
from pyqtgraph.exporters import Exporter, SVGExporter, ImageExporter
from pyqtgraph.parametertree import Parameter,parameterTypes
from pyqtgraph import PlotItem

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


class PlotStyleParameters(parameterTypes.GroupParameter):
    """Parameters for the customized Matplotlib Exporter.
    
    These parameters populate the export UI dialog with options to tweak the style used by matplotlib.
    """
    _selected_styles = []
    def __init__(self,**opts):
        styles = style.available
        if 'default' not in styles:
            styles.insert(0,"default")
        styles = [s for s in styles if not s.startswith("_")]
        if TOOLBOXSTYLE not in styles:
            styles.insert(1,TOOLBOXSTYLE)
        super().__init__(name="Export options",**opts)
        self.addChildren(
            [
                {"name":"dpi","value":144,"type":"int","bounds":(30,2400)},
                {"name":"layout engine","type":"list","limits":["constrained","tight","from style"]},
                {"name":"legend", "type": "group","children":
                    [
                        {"name": "legend names","type":"list","limits":["full","short"]},
                        {"name":"legend location","type":"list","limits":["best","upper left","upper right","lower left","lower right"]},
                        {"name":"legend handle length","type":"float","value":0.5,"bounds":(0.1,2)},
                    ]
                },
                {
                    "name":"Matplotlib styles",
                    "type":"group",
                    "children":[
                        Parameter.create(name=f"{name}",type="bool",value=name in PlotStyleParameters._selected_styles) for name in styles
                    ]
                }
            ]
        )


    def active_styles(self):
        """Figure out the styles that should be applied to the figure.
        
        Multiple styles can be applied, which may override eachother in certain parts.
        """
        selected = [child.name() for child in self.child("Matplotlib styles").children() if child.value() is True]
        PlotStyleParameters._selected_styles = selected
        return selected


class OESMatplotlibExporter(MatplotlibExporter):
    """A customized exporter for pyqtgraph to matplotlib.
    
    It overrides the default (pyqtraph) behaviour slightly to improve the representation.
    
    The user can now set multiple Matplotlib stylesheets to apply to the created plot, at runtime.

    Also, the top and right axes are no longer removed, if they are shown in pyqtgraph as well, and enabled by the chosen stylesheets.

    To aid in legend legibility, it is possible to shorten names, reduce the handle length and shorten names, if needed.
    """
    Name = "Matplotlib (OESToolbox)"

    def __init__(self,item):
        super().__init__(item)
        self.params = PlotStyleParameters()

    @staticmethod
    def make_legend_name(full_name:str,shorten=False) -> str:
        """Abbreviate names for use in the legend, if desired."""
        if not shorten:
            return full_name.strip()
        elif "/" in full_name:
            return full_name.rsplit("/",1)[1].strip()
        else:
            return full_name.rsplit(":",1)[1].strip()
        

    def get_pen_linestyle(self,pen):
        match pen.style():
            case Qt.PenStyle.SolidLine:
                style = "-"
            case Qt.PenStyle.DotLine:
                style  = ":"
            case Qt.PenStyle.DashLine:
                style  = "--"
            case Qt.PenStyle.DashDotLine:
                style = "-."
            case Qt.PenStyle.PenStyle.DashDotDotLine:
                style = "dashdotdotted"
            case Qt.PenStyle.NoPen:
                style ="none"
            case _:
                style="-"
        return {'ls':style,'color':pen.color().getRgbF(),'linewidth':pen.width()}
    
    def get_symbol_style(self,options):
        """Translate pyqtgraph symbol styles to matplotlib compatible keyword arguments."""
        symbolPen = pg.mkPen(options['symbolPen'])
        symbolBrush = pg.mkBrush(options['symbolBrush'])
        markeredgecolor = symbolPen.color().getRgbF()
        markerfacecolor = symbolBrush.color().getRgbF()
        markersize = options['symbolSize']
        return {
            "marker":_symbol_pg_to_mpl.get(options["symbol"], ""),
            "mec":markeredgecolor,
            "mfc": markerfacecolor,
            "ms":markersize}


    
    def parameters(self)->PlotStyleParameters:
        return self.params

    def export(self, fileName=None):
        if not isinstance(self.item,PlotItem):
            QMessageBox.information(
                None,
                f"{self.item.__class__.__name__} not supported",
                f"This exporter only support a `PlotItem`, not `{self.item.__class__.__name__}`.\n"
                "Please select 'Plot' as the target of this export, not 'ViewBox' or 'Entire Scene'."
            )
            return
        mpw = MatplotlibWindow()
        MatplotlibExporter.windows.append(mpw)
        abbreviate = self.params.child("legend")['legend names'].lower() == "short"
        with style.context(self.params.active_styles(), after_reset=True):
            fig = mpw.getFigure()
            if self.params['layout engine'].lower()=="constrained":
                fig.set_constrained_layout(True)
            dpi=self.params['dpi']
            if fig.dpi!=dpi:
                w_px, h_px = fig.canvas.width(), fig.canvas.height()
                fig.set_size_inches(w_px / dpi, h_px / dpi,forward=True)
                fig.set_dpi(dpi)
            xax = self.item.getAxis('bottom')
            yax = self.item.getAxis('left')
            ax_right = self.item.getAxis("right")
            ax_top = self.item.getAxis("top")

            # get labels from the graphic item
            xlabel = xax.label.toPlainText()
            ylabel = yax.label.toPlainText()
            right_label = ax_right.label.toPlainText()
            top_label = ax_top.label.toPlainText()
            title = self.item.titleLabel.text

            # if axes use autoSIPrefix, scale the data so mpl doesn't add its own
            # scale factor label
            xscale = yscale = right_scale=top_scale= 1.0
            if xax.autoSIPrefix:
                xscale = xax.autoSIPrefixScale
            if yax.autoSIPrefix:
                yscale = yax.autoSIPrefixScale
            if ax_right.autoSIPrefix:
                right_scale = ax_right.autoSIPrefixScale
            if ax_right.autoSIPrefix:
                top_scale = ax_top.autoSIPrefixScale

            ax = fig.add_subplot(111, title=title)
            ax.clear()
            for item in self.item.curves:
                if not item.isVisible():
                    continue
                x, y = item.getData()
                item_label = self.make_legend_name(item.name(), abbreviate)
                x = x * xscale
                y = y * yscale

                opts = item.opts
                pen = pg.mkPen(opts['pen'])
                line_style = self.get_pen_linestyle(pen)
                marker_style = self.get_symbol_style(opts)
                
                if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
                    fillBrush = pg.mkBrush(opts['fillBrush'])
                    fillcolor = fillBrush.color().getRgbF()
                    ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
                
                ax.plot(x, y, **line_style,**marker_style, label=item_label, zorder=item.zValue())

                xr, yr = self.item.viewRange()
                ax.set_xbound(xr[0]*xscale, xr[1]*xscale)
                ax.set_ybound(yr[0]*yscale, yr[1]*yscale)
                

            ax.set_xlabel(xlabel)  # place the labels.
            ax.set_ylabel(ylabel)
            # Don't force ticks (by line below), but let this be controlled by the selected style(s).
            # ax.tick_params(axis='both', which='both',direction='in',bottom=True, top=True,left=True,right=True, labelleft=True,labelbottom=True, labelright=False,labeltop=False)
            if self.item.legend.isVisible() and (len(self.item.legend.items)>0):
                ax.legend(loc=self.params.child("legend")["legend location"], handlelength=self.params.child("legend")["legend handle length"])
            if self.params['layout engine'].lower()=="tight":
                fig.tight_layout()
            mpw.draw()

class OESDataExporter(Exporter):
    """An exporter that exports graphed data from the OES-toolbox to a file (text, excel, parquet).
    
    This exporter is intended to be used instead of the default pyqtgraph CSVExporter/HDF5Exporter, to be consistent with other methods to export in the OES-toolbox.

    In essence, it triggers the appropriate export action from the main toolbox window, depending on the user selection.

    In case of using a `Plot data` export, the resulting file can be read/plotted again with the OES-toolbox as a set of spectra.
    """
    Name = "Export data (OES-toolbox)"
    
    def __init__(self,item:PlotItem|GraphicsScene):
        super().__init__(item)
        self.params = Parameter.create(name='params', type='group', children=[
            {"name":"Export data","type":"list","limits":["Plot data","NIST table","Molecule fit results","Continuum results"]},
        ])
        self.main_window = item.getViewWidget().window()


    def export(self):
        """Export requested data by calling the appropriate export action on the toolbox main window."""
        kind = self.params.child("Export data").value()
        match kind:
            case "Plot data":
                self.main_window.action_export_plot_data.trigger()
            case _s if "NIST" in _s:
                self.main_window.action_export_ident_table.trigger()
            case _s if "Molecule" in _s:
                self.main_window.action_export_molecule_fit_results.trigger()
            case _s if "Continuum" in _s:
                self.main_window.action_export_continuum_fit_results.trigger()
            case _:
                QMessageBox.warning(self.main_window, "Failed to export file",f"Could perform requested export: {kind}")

    def parameters(self):
        return self.params
    
# Clear the default exports, then register the ones we want to expose in order of priority.
# SVG/ImageExporter may still be usefull for users.
pg.exporters.Exporter.Exporters.clear()
OESMatplotlibExporter.register()
OESDataExporter.register()
SVGExporter.register()
ImageExporter.register()