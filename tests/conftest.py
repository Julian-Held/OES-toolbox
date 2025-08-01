import pytest
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import sampled_from, lists, floats, text
from pathlib import Path
import numpy as np
import pandas as pd
import sif_parser

@pytest.fixture(scope="session")
def horiba_file():
    """Sample file provided for issue #4."""
    file = Path("./tests/test_files/Horiba.txt")
    df = pd.read_csv(file, skiprows=21, header=None,sep="\t",skipfooter=638-322, engine="python").dropna(axis=1).T.astype(float)
    break_next = False
    with file.open("r") as fo:
        for line in fo:
            if break_next is True:
                break
            if "HRes Wavelength" in line:
                break_next = True
    wavelength = np.fromstring(line,sep="\t")        
    
    yield wavelength,df

@pytest.fixture(scope="session")
def Andor_kinetic_sif_file():
    """A kinetic series of the C2 Swan bands recorded with Andor Solis and saved as an SIF file.

    When using `sif_parser` version 0.3.5 or older, the wavelength calibration will be missing or invalid, thus we read it ourselves.

    This method will not work for step-and-glue spectra.
    """
    file = Path("./tests/test_files/Andor_kinetic.sif")
    data = sif_parser.xr_open(file)
    if "calibration" not in data.coords:
        with file.open("rb") as fo:
            for _ in range(50):
                if fo.readline().startswith(b"65539"):
                    calib = np.flip(list(map(float, fo.readline().split())))
                    break
        data = data.assign_coords(calibration=("width", np.polyval(calib, np.arange(1,data.ImageLength+1))))
    yield data


@pytest.fixture(scope="session")
def Avantes_raw8_demo_file():
    """A sample file recorded by AvaSoft 8 running in demo mode."""
    file = Path("./tests/test_files/avasoft8_demo.raw8")
    pixel_indices = np.fromfile(file,dtype=np.dtype("H"),offset=89,count=2)
    data = pd.DataFrame(
        np.fromfile(
            file,
            dtype=np.dtype("<f"),
            offset=328,
            count=(np.diff(pixel_indices)+1)[0]*4
        ).reshape(4,-1).T.astype(float),
        columns=['wl','scope', 'dark','ref']
    )
    yield data


@pytest.fixture(scope="session")
def example_dataframe():
    rng = np.random.default_rng()
    demo_df = pd.DataFrame({"wavelength":np.linspace(100,1000,50), **{str(i):rng.random(50) for i in range(10)}})
    yield demo_df

@pytest.fixture(scope="session")
def temp_text_files(tmp_path_factory, example_dataframe):
    """Create a set of temporary files of various encodings and different delimiter/decimal characters to test against."""
    tmp_path = tmp_path_factory.mktemp("test_text_files")
    for encoding in ["utf-8","ascii","cp1252", "utf-16","utf-16be","utf-16le","utf-32","macroman"]:
        for name,sep,dec in zip(
            ["comma_dot","tab_dot","tab_comma","semicolon_dot","semicolon_comma","bar_dot","bar_comma","space_dot","space_comma"],
            [",","\t","\t",";",";","|","|"," ", " "],
            [".",".",",", ".",",",".",",", ".", ","],
            strict=True
        ):
            fname = f"{name}_{encoding}.txt"
            with tmp_path.joinpath(fname).open("w", encoding=encoding) as fo:
                fo.write("# Text at top of file\n")
                fo.write("This serves to simulate a block of metadata before actual data starts\n")
                fo.write("It should be ignored regardless of a line starting with a # as comment character\n")
            example_dataframe.to_csv( tmp_path.joinpath(fname), sep=sep, decimal=dec, index=False, mode="a", encoding=encoding)
    yield tmp_path
