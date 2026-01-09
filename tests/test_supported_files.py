"""A test suite running on sample files of formats that we (attempt to) support.

This should help find breaking changes on our end or in upstream dependencies.

The different test cases are based on sample files that were actually recorded by various software programs.

They are stored in `./test_files`.
"""

import pytest
from pathlib import Path
from OES_toolbox.file_handling import FileLoader
import numpy as np
import pandas as pd
import sif_parser

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xarray import DataArray

from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_index_equal
from xarray.testing import assert_identical

class TestSupportedFiles:
        """Class containing test cases to test supported file types with concrete example files.
        
        Tests files that we handle ourselves, and those for which we have a dependency.

        This enables us to detect if file support breaks even if changes happen upstream.

        For each example file, specific information (e.g. shape, sum, etc.) is hardcoded in the tests to validate consistency.
        """


        def test_read_horiba_txt(self,horiba_file):
            """Test the file provided as a sample for issue #4."""
            wl,data,_ = FileLoader.read_horiba_txt("./tests/test_files/Horiba.txt")
            assert wl.shape == (2048,)
            assert data.shape == (2048,300)
            assert data.shape[0] == wl.shape[0]
            assert_index_equal(data.index,pd.RangeIndex(2048))
            assert_allclose([wl.min(),wl.mean(),wl.max()],[190.037,540.8279,884.273])
            assert_allclose([data.mean().min(),data.mean().mean(),data.mean().max()],[4491.5849,4703.6766,66363.9409])
            assert_frame_equal(data,horiba_file[1])
            assert_allclose(wl,horiba_file[0])

        def test_read_andor_sif_numpy(self,Andor_kinetic_sif_file:"DataArray"):
            """Test a sample Andor SIF file containing a kinetic series, using `sif_parser` in `numpy` mode.

            For `sif_parser` version 0.3.5 and earlier this relies on our custom extraction the wavelength calibration; this will not work for step-and-glue.
            """
            wl,data = FileLoader.read_andor_sif(Path("./tests/test_files/Andor_kinetic.sif"))
            assert data.shape == (100,1,1024)
            assert_allclose([wl.min(), wl.mean(),wl.max()],[452.4057,487.0280,521.4877])
            assert_allclose([data.min(),data.mean(),data.max()],[0,19029.64,186700],atol=0.004) # on py 3.10 an 3.11 mean will be 19029.637
            assert_allclose(data,Andor_kinetic_sif_file.data)

        def test_read_andor_sif_xarray(self,Andor_kinetic_sif_file:"DataArray"):
            """Test a sample Andor SIF file containing a kinetic series, using `sif_parser` in `xarray` mode.

            For `sif_parser` version 0.3.5 and earlier this relies on our custom extraction the wavelength calibration; this will not work for step-and-glue.
            """
            data =FileLoader.read_andor_sif_xarray(Path("./tests/test_files/Andor_kinetic.sif"))
            assert data.shape == (100,1,1024)
            assert data.size == 102400
            assert data.attrs['size'] == (1024,1)
            assert data.shape == (100,1,1024)
            assert data.dims == ('Time', 'height', 'width')
            assert data.attrs['SifVersion'] == 65567
            assert data.calibration.dims == ('width',)
            assert_allclose([data.calibration.min(),data.calibration.mean(),data.calibration.max()],[452.4057,487.0280,521.4877])
            assert_allclose([data.min(),data.mean(),data.max()],[0,19029.64,186700],rtol=2e-7) # on py 3.10 an 3.11 mean() will be 19029.637, meaning sub 2e-7 relative difference
            assert_allclose(data.Time,np.zeros(100))
            assert_identical(data,Andor_kinetic_sif_file)

        def test_read_avantes_raw8(self,Avantes_raw8_demo_file):
            """Read a file created with AvaSoft 8 with a 'virtual demo spectrometer' (e.g. with no physical device attached.)"""
            data = FileLoader.read_avantes_binary(Path("./tests/test_files/Avasoft8_demo.raw8"))
            assert len(data)==1 # contains 1 channel
            assert data[0].ID.SerialNumber == '00000000'
            assert data[0].data.shape == (1615,4) # 4 vectors, wavelength, signal,dark,ref
            assert data[0]==data.channels[0]
            assert round(data[0].exposure,8)==1.04999995
            assert data[0].wavelength.min()==174.029
            assert data[0].wavelength.max()==1100.3231
            assert_allclose(data[0].data,Avantes_raw8_demo_file)
