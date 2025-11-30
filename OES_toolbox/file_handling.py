""""Core module for file handling.

Whenever a new file-type needs to be supported, this is the place to start.
"""

from pathlib import Path
import time
import re
from PyQt6.QtGui import QImage
from matplotlib.figure import Figure
import sif_parser
from OES_toolbox import pyAvantes
import numpy as np
# import pandas as pd
# import xarray as xr
from charset_normalizer import is_binary, from_bytes, from_fp
# from spexread.parsing import read_spe_file
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray,Dataset
    from numpy.typing import NDArray,ArrayLike

from OES_toolbox.logger import Logger
from OES_toolbox.lazy_import import lazy_import

pd = lazy_import("pandas")
xr = lazy_import("xarray")
spexread = lazy_import("spexread")


class SpectraDataset:
    """A dataset of spectra recorded with the same wavelength axis and/or region of interest.

    The aim of this class is to provide a consistent interface for spectra read from any file.

    Attributes:
        x (ArrayLike, 1D)   :   The horizontal/wavelength axis of the spectrum
        y (ArrayLike, ND)   :   The set of spectra recorded for `x`.
    """

    def __init__(self,x:np.typing.ArrayLike,y:np.typing.ArrayLike, background:None|np.typing.ArrayLike=None, name:str="spectrum"):
        self.x = x if isinstance(x,np.ndarray) else x.to_numpy()
        self.y = y if isinstance(y,np.ndarray) else y.to_numpy()
        if np.ndim(self.y)>1:
            self.y = self.y.reshape(-1) if np.shape(self.y)[1]<2 else self.y
        self.background = background if background is not None else np.zeros_like(x)
        self.background = self.background if isinstance(self.background, np.ndarray) else self.background.to_numpy()
        self.has_background = not np.array_equal(self.background,np.zeros_like(self.x))
        self.name = name

    @property
    def shape(self):
        return (self.y.shape)
    
    def __repr__(self):
        return f"SpectraDataset(name={self.name}, shape={self.shape}, has_background={self.has_background})"

class FileLoader:
    """The core class responsible for loading data.
    
    Contains various methods and helpers to facilitate inferring file schema and loading the data.
    """
    logger = Logger(instance=None, context={"class":"FileLoader"})

    @staticmethod
    def _infer_text_schema_from_line(line: str) -> tuple[str, str]:
        """Naively tries to infer the schema of a file from a single line.
        
        Only supports a small set of common column delimiters and decimal characters.
        """
        delimiters = ["\t", ";", "|", " ",","] # put comma last to priorize the others due to ambiguity based on locale
        delim = next(d for d in delimiters if d in line)
        decimal_chars = [char for char in [",", "."] if char != delim]
        dec = next((d for d in decimal_chars if d in line.replace(delim, "")), ".")
        return delim, dec
    
    @staticmethod
    def _parse_open_text_file(handle,offset,sep:str=",",decimal:str=".",names: list|None=None, **kwargs)-> pd.DataFrame:
        """Read the data block from an already opened text file.
        
        Will drop all columns that are filled with NaN values and cast all data to float.
        
        Designed to only reading the data block (e.g. numeric data), no header etc.

        Additional kwargs will be passed to `pandas.read_csv`, to customize read behaviour.
        
        Args:
            handle  : the handle of the opened file object
            offset  : The offset to start reading from.
            sep     : The separator/delimiter of the data
            decimal : The decimal character
            names   : Optional names for the columns.
        """
        header = kwargs.pop('header', None)
        engine = kwargs.pop('engine', 'pyarrow')        
        current_pos = handle.tell()
        handle.seek(offset)
        df = pd.read_csv(handle,sep=sep,decimal=decimal,names=names, header=header, engine=engine, **kwargs)
        handle.seek(current_pos)
        return df.dropna(axis=1,how='all').astype(float)
    
    @classmethod
    def _read_generic_text(cls, f: Path) -> pd.DataFrame:
        """Infer the format and read from a generic text file.
        
        Intedend as a catch-all reader for any text format that does not require bespoke parsing.
        """
        f = Path(f)
        with f.open("rb") as fo:
            try:
                enc = from_fp(fo).best().encoding
            except AttributeError as err:
                cls.logger.warning("Could not detect encoding for '%s', perhaps this is a binary file (tried generic text file).",f.name)
                raise EncodingWarning(f"Could not detect encoding for '{f.name}', perhaps this is a binary file (tried generic text file).") from err
        with f.open("r", encoding=enc) as fo:
            pos = []
            line_num = 0
            while line_num < 50:
                cursor = fo.tell()
                pos.append(cursor)
                line = fo.readline().strip()
                if line and line[0].isdigit():  # first line with data
                    sep, decimal = cls._infer_text_schema_from_line(line)
                    offset_data = cursor
                    line_num_data = line_num
                    break
                line_num += 1
            cls.logger.debug(f"{f.name}: {enc}, {sep=}, {decimal=}, {offset_data=}")
            df = cls._parse_open_text_file(fo,offset_data,sep,decimal, on_bad_lines='skip')
            # determine column names if any
            fo.seek(pos[pos.index(offset_data)-1])
            heading_line = fo.readline().strip()
            if heading_line =="":
                fo.seek(pos[pos.index(offset_data)-2])
                heading_line = fo.readline().strip()
            
            if sep in heading_line:
                names = [part.strip() for part in heading_line.split(sep) if part.strip()!=""]
                if (len(names)==df.shape[1]) & (np.unique(names).shape[0]==len(names)):
                    df.columns = names
            return df

    @classmethod
    def read_avantes_txt(cls, f: Path) -> pd.DataFrame:
        data = cls._read_generic_text(f)
        return data

    @classmethod
    def read_avantes_raw8(cls, f: Path) -> pd.DataFrame:
        data = pd.DataFrame(pyAvantes.Raw8(f).data)
        return data

    @classmethod
    def read_andor_sif(cls, f: Path)->tuple["NDArray","NDArray"]:
        """Read a `*.sif` file such as created by Andor SOLIS software, using the `sif_parser` package, into a numpy arrays.
        
        Requires a more recent version of `sif_parser` than 0.3.5 to properly extract wavelength axis for step-and-glue or older formats.

        However, contains a crude patch to retrieve the calibration for version 0.3.5 for some files
        """
        # data = sif_parser.xr_open(f)
        data, meta = sif_parser.np_open(f)
        wl = sif_parser.utils.extract_calibration(meta)
        if (wl is None) or (np.shape(data)[-1]!=np.shape(wl)[-1]):
            with f.open("rb") as fo:
                for _ in range(50):
                    if fo.readline().startswith(b"65539"):
                        calib = np.flip(list(map(float, fo.readline().split())))
                        break
            wl = np.polyval(calib,np.arange(1,meta['ImageLength']+1))

        # print(f"{np.shape(data)=} {np.shape(wl)=}, {meta['ImageLength']}")
        return wl,data
    
    @classmethod
    def read_andor_sif_xarray(cls, f: Path)->"DataArray":
        """Read a `*.sif` file such as created by Andor SOLIS software, using the `sif_parser` package, into an xarray DataArray.
        
        Requires a more recent version of `sif_parser` than 0.3.5 to properly extract wavelength axis for step-and-glue or older formats.

        However, contains a crude patch to retrieve the calibration for version 0.3.5 for some files
        """
        data = sif_parser.xr_open(f)
        if "calibration" not in data.coords:
            with f.open("rb") as fo:
                for _ in range(50):
                    if fo.readline().startswith(b"65539"):
                        calib = np.flip(list(map(float, fo.readline().split())))
                        break
            data = data.assign_coords(calibration=("width", np.polyval(calib, np.arange(1,data.ImageLength+1))))
        return data

    @classmethod
    def read_andor_asc(cls, f: Path) -> pd.DataFrame:
        data = cls._read_generic_text(f)
        return data

    @classmethod
    def read_PI_spe(cls, f: Path)->list["DataArray"]:
        return spexread.parsing.read_spe_file(f,as_dataset=False)

    @classmethod
    def read_netCDF(cls, f:Path):
        return xr.load_dataset(Path(f))
        
    
    @classmethod
    def read_horiba_txt(cls,f:Path) -> tuple["ArrayLike",pd.DataFrame]:
        f = Path(f)
        time_block_start = 0
        sep = "\t" # assumed constant
        decimal = "." # assume data in of type int, thus does not matter
        with Path(f).open('rb') as fo:
            enc = from_fp(fo).best().encoding
            fo.seek(0)
            for line in fo:
                if b'Wavelength' in line:
                    wavelength_block_start = fo.tell()
                elif b'Raw Intensity' in line:
                    data_block_start = fo.tell()
                elif (line.startswith(b'[')) & (time_block_start==0):
                    time_block_start = fo.tell()
                    break # done with finding blocks in file
            fo.seek(wavelength_block_start)
            wavelength = np.fromstring(fo.readline().decode(enc),sep= sep)
            y = cls._parse_open_text_file(fo,data_block_start,sep=sep,decimal=decimal, on_bad_lines="skip").T
            expr = re.compile(r"\[([\d\.\,]+)\]")
            fo.seek(time_block_start)
            timestamps = np.unique(np.fromiter(map(float, expr.findall(fo.read(-1).decode(enc))), float))
        cls.logger.debug(f"{f.name}: {data_block_start=},{wavelength_block_start=}, {time_block_start=}")
        return wavelength,y, timestamps
        


    @classmethod
    def open_any_spectrum(cls, f: Path, sample_size=1024) -> list[SpectraDataset]:
        f = Path(f)
        time_start = time.perf_counter()
        with f.open("rb") as fo:
            sample: bytes = fo.read(sample_size)
        is_bin:bool = is_binary(sample)
        ext = f.suffix.lower()
        if b"Data measured with spectrometer [name]:" in sample:
            data = cls.read_avantes_txt(f)
            spectra = [SpectraDataset(x=data.iloc[:, 0],y=data.iloc[:, 1:])]
        elif b"AVS" in sample[:3]:
            data = cls.read_avantes_raw8(f)
            spectra = [SpectraDataset(x=data.iloc[:, 0],y=data.loc[:, 'scope'], background=data.loc[:,'dark'].to_numpy())]
        elif b"Andor Technology Multi-Channel File" in sample:
            data = cls.read_andor_sif(f)
            spectra = [SpectraDataset(x=data[0],y=data[1].sum(axis=1).T)]
        elif is_bin & (ext == ".spe"):
            data:xr.Dataset = cls.read_PI_spe(f)
            spectra = []
            for d in data:
                other_dim = [name for name in ['x','y'] if name not in d.wavelength.dims]
                spectra.append(SpectraDataset(x=d.wavelength,y=d.mean(other_dim).data.T, name=d.name))
        elif b"OESi Camera" in sample:
            x,y,_ = cls.read_horiba_txt(f)
            spectra = [SpectraDataset(x=x,y=y)]
        elif is_bin & (ext=='.nc'):
            data = cls.read_netCDF(f)
            spectra = []
            for name,roi in data.items():
                d = roi.sum([dim for dim in roi.dims if dim not in [roi.wavelength.dims[0], roi.time.dims[0]]])
                bg = getattr(roi,'background',None)
                spectra.append(SpectraDataset(x=roi.wavelength,y=d.T, background=bg, name=name))
        else:
            data = cls._read_generic_text(f)
            x = data.iloc[:, 0].to_numpy()
            y = data.iloc[:, 1:].to_numpy()
            spectra = [SpectraDataset(x=x,y=y)]
        cls.logger.info(f"Read '{f.name}' (size: {f.stat().st_size*1e-6:.3f} MB) in {(time.perf_counter()-time_start)*1e3:.2f} ms.")
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
                style_kws["label"]=plot_item.name().strip("file:").strip()
            if plot_item.name().startswith("cont"):
                style_kws["lw"] = 1
            elif plot_item.name().startswith('NIST'):
                style_kws["lw"] = 0.5
            x,y = plot_item.getData()
            plt.plot(x,y,**style_kws)
            count += 1
        plt.xlim(xlim)
        plt.ylim(ylim)
        # if count>1:
        if graph.plotItem.legend.isVisible():
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
