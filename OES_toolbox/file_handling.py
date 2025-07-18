""""Core module for file handling.

Whenever a new file-type needs to be supported, this is the place to start.
"""

from pathlib import Path
import importlib.util
import time
from PyQt6.QtGui import QImage
from matplotlib.figure import Figure
import sif_parser
from OES_toolbox import pyAvantes
import numpy as np
import pandas as pd
import xarray as xr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray,Dataset

from OES_toolbox.logger import Logger



class SpectraDataset:
    """A dataset of spectra recorded with the same wavelength axis and/or region of interest.

    The aim of this class is to provide a consistent interface for spectra read from any file.

    Attributes:
        x (ArrayLike, 1D)   :   The horizontal/wavelength axis of the spectrum
        y (ArrayLike, ND)   :   The set of spectra recorded for `x`.
    """

    def __init__(self,x:np.typing.ArrayLike,y:np.typing.ArrayLike, background:None|np.typing.ArrayLike=None):
        self.x = x if isinstance(x,np.ndarray) else x.to_numpy()
        self.y = y if isinstance(y,np.ndarray) else y.to_numpy()
        self.background = background if background is not None else np.zeros_like(x)
        self.background = self.background if isinstance(self.background, np.ndarray) else self.background.to_numpy()
        self.has_background = not np.array_equal(self.background,np.zeros_like(self.x))

    @property
    def shape(self):
        return (self.y.shape)
    
    def __repr__(self):
        return f"SpectraDataset(shape={self.shape}, has_background={self.has_background})"

class FileLoader:
    """The core class responsible for loading data.
    
    Contains various methods and helpers to facilitate inferring file schema and loading the data.
    """
    logger = Logger(instance=None, context={"class":"FileLoader"})

    @staticmethod
    def is_binary_string(buffer: bytes) -> bool:
        """Checks if string a binary buffer contains binary data, by stripping all ASCII and many control characters."""
        textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
        return bool(buffer.translate(None, textchars))

    @staticmethod
    def _infer_text_schema_from_line(line: str) -> tuple[str, str]:
        delimiters = ["\t", ";", "|", ","]
        delim = next(d for d in delimiters if d in line)
        decimal_chars = [char for char in [",", "."] if char != delim]
        dec = next((d for d in decimal_chars if d in line.replace(delim, "")), ".")
        return delim, dec

    @classmethod
    def _read_generic_text(cls, f: Path, verbose=False) -> pd.DataFrame:
        pos = [0]
        line_num = 0
        header = None
        names = None
        with f.open("r") as fo:
            while line_num < 50:
                line = fo.readline().strip()
                if len(line) > 0:  # noqa: SIM102
                    if line[0].isdigit():  # first line with data
                        delim, dec = cls._infer_text_schema_from_line(line)
                        break
                pos.append(fo.tell())
                line_num += 1
            # determine header
            if line_num == 0:
                goto = 0
            else:
                goto = pos[-2]
                fo.seek(goto)
                raw = fo.readline()
                if raw.strip() == "":
                    fo.seek(pos[-4])
                    raw = fo.readline()
                    goto = pos[-1]
                if delim not in raw:
                    header = None
                    names = None
                    goto = pos[-1]
                else:
                    header = 0 if goto == pos[-2] else None
                    names = [s.strip() for s in raw.split(delim)]
                    if np.unique(names).shape[0] < len(names):
                        names = None
            fo.seek(goto)

            if verbose:
                print(f"{delim=}, {dec=},{line_num=},{goto=},{header=}, {names=}")
            try:
                df = pd.read_csv(
                    fo, sep=delim, decimal=dec, header=header, names=names, engine="pyarrow", on_bad_lines="error"
                )
            except pd.errors.ParserError:
                # Fallback to the `C` parser when `pyarrow` fails
                fo.seek(goto)
                df = pd.read_csv(fo,sep=delim, decimal=dec, header=header, names=names,engine='c')
        return df.dropna(axis=1, how="all").dropna(axis=0, ignore_index=True)

    @classmethod
    def read_avantes_txt(cls, f: Path, verbose=False) -> pd.DataFrame:
        data = cls._read_generic_text(f, verbose=verbose)
        return data

    @classmethod
    def read_avantes_raw8(cls, f: Path) -> pd.DataFrame:
        data = pd.DataFrame(pyAvantes.Raw8(f).data)
        return data

    @classmethod
    def read_andor_sif(cls, f: Path)->"DataArray":
        data = sif_parser.xr_open(f)
        if "calibration" not in data.coords:
            with f.open("rb") as fo:
                for _ in range(50):
                    if fo.readline().startswith(b"65539"):
                        calib = np.flip(list(map(float, fo.readline().split())))
            data = data.assign_coords(calibration=("width", np.polyval(calib, data.width)))
        return data

    @classmethod
    def read_andor_asc(cls, f: Path, verbose=False) -> pd.DataFrame:
        data = cls._read_generic_text(f, verbose=verbose)
        return data

    @classmethod
    def read_PI_spe(cls, f: Path)->"Dataset":
        raise NotImplementedError

    @classmethod
    def read_netCDF(cls, f:Path):
        return xr.load_dataset(Path(f))
        
    
    @classmethod
    def read_horiba_txt(cls,f:Path) -> tuple[np.typing.ArrayLike,pd.DataFrame]:
        line_num=0
        data_block_end = 0
        time_block_start = 0 
        with Path(f).open('rb') as fo:
            for i,line in enumerate(fo):
                if b'Wavelength' in line:
                    wavelength_block_start = fo.tell()
                elif b'Raw Intensity' in line:
                    data_block_start = fo.tell()
                elif (line.startswith(b"*")) & (data_block_end==0):
                    data_block_end = fo.tell()
                    data_block_end_line = i
                elif (line.startswith(b'[')) & (time_block_start==0):
                    time_block_start = fo.tell()
                line_num += 1
            last_line = i+1
            fo.seek(wavelength_block_start)
            wavelength = np.fromstring(fo.readline().decode(),sep='\t')
            fo.seek(data_block_start)
            y = pd.read_csv(
                fo,
                sep='\t', 
                engine='python', 
                skipfooter=last_line-data_block_end_line,
                header=None
                ).dropna(axis=1).T
            
        return wavelength,y
        


    @classmethod
    def open_any_spectrum(cls, f: Path, sample_length=1024, verbose=False) -> list[SpectraDataset]:
        f = Path(f)
        time_start = time.perf_counter()
        with f.open("rb") as fo:
            sample: bytes = fo.read(sample_length)
        is_bin = cls.is_binary_string(sample)
        ext = f.suffix
        if b"Data measured with spectrometer [name]:" in sample:
            data = cls.read_avantes_txt(f, verbose=verbose)
            spectra = [SpectraDataset(x=data.iloc[:, 0],y=data.iloc[:, 1:])]
        elif b"AVS" in sample[:3]:
            data = cls.read_avantes_raw8(f)
            spectra = [SpectraDataset(x=data.iloc[:, 0],y=data.iloc[:, 1:])]
        elif b"Andor Technology Multi-Channel File" in sample:
            data = cls.read_andor_sif(f)
            x = data.calibration.to_numpy()
            spectra = [SpectraDataset(x=x,y=data.data.T.reshape(x.shape[0], -1))]
        elif is_bin & (ext == ".spe"):
            raise NotImplementedError()
            data:xr.Dataset = None
            dim_name_wavelength = data.wavelength.dims[0]
            dim_name_others = [name for name in data.dims if name not in data.wavelength.dims]
            spectra = []
            for name, roi in data.items():
                d = roi.dropna(dim_name_wavelength,how='all').mean(dim_name_others)
                spectra.append(SpectraDataset(x=d.wavelength,y=d.data))
        elif b"OESi Camera" in sample:
            x,y = cls.read_horiba_txt(f)
            spectra = [SpectraDataset(x=x,y=y)]
        elif is_bin & (ext=='.nc'):
            data = cls.read_netCDF(f)
            spectra = []
            for name,roi in data.items():
                d = roi.sum([dim for dim in roi.dims if dim not in [roi.wavelength.dims[0], roi.time.dims[0]]])
                spectra.append(SpectraDataset(x=roi.wavelength,y=d.T))
        else:
            data = cls._read_generic_text(f, verbose=verbose)
            x = data.iloc[:, 0].to_numpy()
            y = data.iloc[:, 1:].to_numpy()
            spectra = [SpectraDataset(x=x,y=y)]
        cls.logger.debug(f"Opening file took {(time.perf_counter()-time_start)*1e3:.2f} ms")
        return spectra


class FileExport:

    @staticmethod
    def graph_to_matplotlib(graph) -> Figure:
        import matplotlib.pyplot as plt
        from PyQt6.QtCore import Qt
        plt.style.use(Path(__file__).parent.joinpath('ui/jh-paper.mplstyle'))
        xlim,ylim = graph.getViewBox().viewRange()
        fig = plt.figure()
        count = 0
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
                style_kws["label"]="measurement"
            elif plot_item.name().startswith("cont"):
                style_kws["lw"] = 1
            elif plot_item.name().startswith('NIST'):
                style_kws["lw"] = 0.5
            x,y = plot_item.getData()
            plt.plot(x,y,**style_kws)
            count += 1
        plt.xlim(xlim)
        plt.ylim(ylim)
        if count>1:
            plt.legend()
        plt.xlabel("Wavelength / nm")
        plt.ylabel("Intensity")
        return fig
    
    @staticmethod
    def matplotlib_to_image(fig) -> QImage:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from PyQt6.QtGui import QImage, QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = QImage(
            canvas.buffer_rgba(), 
            int(fig.figbbox.width),
            int(fig.figbbox.height),
            QImage.Format.Format_RGBA8888_Premultiplied
        )
        return img
