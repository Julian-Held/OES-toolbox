"""Tests for fitting data/molecular spectra to validate if the fitting logic produces consistent and expected results.

This helps in the troubleshooting and validation process of the fitting logic without relying on the toolbox UI.
"""

from pathlib import Path

import lmfit
import Moose
import numpy as np
import pytest

from OES_toolbox.calc import make_fit_params, molecule_objective
from OES_toolbox.file_handling import FileLoader

db_CN = Moose.query_DB("CNBX")
db_N2 = Moose.query_DB("N2CB")


def test_N2_CN_spectrum():
    """A cannonical test case for fitting based on a experimental spectrum of CN Violet and N2 Second Positive systems.
    
    This test is to validate OES-toolbox remains consistent in the approach to fitting; providing the same results for key parameters.

    If molecule fitting logic is touched, this helps detect if there were unintended side-effects.

    Only adapt this test if there are clear reason to do so!

    The nice thing about this sample spectrum are:
     - It contains clear signatures of several species, so we can test fitting of several species
     - The source data contains NaN values, which need to be filtered out.
    """
    data = FileLoader.open_any_spectrum(Path(__file__).parent.joinpath("test_files/370-400nm_9ns_calib.txt"))[0]

    x = data.x
    y = data.y[:,0]
    

    params = lmfit.create_params(**Moose.default_params)
    params.pop("T_rot")
    params.pop("T_vib")
    params.add("T_rot_CNBX",value=3000,min=300,max=15000)
    params.add("T_rot_N2CB",value=3000,min=300,max=15000)
    params.add("T_vib_CNBX",value=3000,min=300,max=15000)
    params.add("T_vib_N2CB",value=3000,min=300,max=15000)
    params.add("fraction_CNBX",value=0.5,min=0,max=1)
    params.add("fraction_N2CB",value=0.5,min=0,max=1)
    # NaN values in data will raise ValueError at first
    with pytest.raises(ValueError,match="NaN values detected"):
        result = lmfit.minimize(molecule_objective, params, args=(x, y), kws={"CNBX":db_CN, "N2CB":db_N2})

    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]
    result = lmfit.minimize(molecule_objective, params, args=(x, y), kws={"CNBX":db_CN, "N2CB":db_N2})

    rtol = 1e-3 # relative tolerance
    assert result.params['T_rot_CNBX'].value == pytest.approx(2009.2,rel=rtol)
    assert result.params['T_rot_N2CB'].value == pytest.approx(573.6,rel=rtol)
    assert result.params['T_vib_CNBX'].value == pytest.approx(9364,rel=rtol)
    assert result.params['T_vib_N2CB'].value == pytest.approx(3707,rel=rtol)
    assert result.params['fraction_CNBX'].value == pytest.approx(4.945e-2,rel=rtol)
    assert result.params['fraction_N2CB'].value == pytest.approx(1.923e-3,rel=rtol)

    # Repeat with axis scaling enabled; fit params will change somewha
    params_scale = result.params.copy()
    params_scale.add("lambda_first_order", value=1, min=0.5, max=2)
    params_scale.add("lambda_second_order", value=0, min=-0.5, max=0.5)

    result = lmfit.minimize(molecule_objective, params_scale, args=(x, y), kws={"CNBX":db_CN, "N2CB":db_N2})
    assert result.params['lambda_first_order'].value == pytest.approx(1,rel=rtol)
    assert result.params['lambda_second_order'].value == pytest.approx(0,abs=rtol)
    assert result.params['T_rot_CNBX'].value == pytest.approx(2020,rel=rtol)
    assert result.params['T_rot_N2CB'].value == pytest.approx(583.6,rel=rtol)
    assert result.params['T_vib_CNBX'].value == pytest.approx(9373,rel=rtol)
    assert result.params['T_vib_N2CB'].value == pytest.approx(3741,rel=rtol)
    assert result.params['fraction_CNBX'].value == pytest.approx(4.957e-2,rel=rtol)
    assert result.params['fraction_N2CB'].value == pytest.approx(1.949e-3,rel=rtol)