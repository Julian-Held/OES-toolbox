"""Module for calculations (and their helper functions) performed by OES toolbox.

These may be thin wrappers around external libraries to provide a consistent, or simplified, API.

Also, they enable us to isolate the toolbox itself from API changes upstream.

When upstream changes, only functions in here need to be modified, rather than code throughout the toolbox.

Finally, these functions can be tested, so we can have some confidence in their correctness (or the flaws in our assumptions...).
"""
from typing import TYPE_CHECKING

import Moose.lmfit
import numpy as np
import pandas as pd
from Moose.utils.caching import array_cache

from .lazy_import import lazy_import

if TYPE_CHECKING:
    from lmfit import Parameters
    from lmfit.minimizer import MinimizerResult

lmfit = lazy_import("lmfit")

PARAMARGS = ('vary',"min",'max')

DEFAULT_PARAMS = Moose.default_params
DEFAULT_PARAMS["T_vib"]['max'] = 25000
DEFAULT_PARAMS["T_rot"]['max'] = 25000


def make_fit_params(y, species:list[str]|dict[str,pd.DataFrame], Trot:float, Tvib:float, sigma:float,gamma:float, mu:float, separate_Tvib=True, separate_Trot=True, vary_broadening=True, vary_shift=False, vary_stretch=False) ->"Parameters":
    """Construct suitable fit parameters with bounds for fitting a spectrum with the `Moose.lmfit.multi_species_objective` function.

    For each element in the `species` list, it will add `fraction` and optional `T_rot`/`T_vib` parameters, as appropriate.
    
    Will calculate bounds and set parameters as free/fixed depending on provided arguments, which can come from the UI state.

    The parameter bounds are made assuming that `multi_species_objective` will be called with the `normalize` kwargs set to `True`.

    This means that `fraction` should be interpreted as a non-normalized `weight` to the total intensity, and is affected by strength of emitter.

    (Stronger emitters will cause a lower weight).

    Arguments:
        y (NDArray):            The array of y-data, to calculate offset parameter `b` and total amplitude `A` from.
        species (list[str]):    List of species names, for the species that will be used in the model objective function
        Trot (float):           Initial estimate of T_rot
        Tvib (float):           Initial estimate of T_vib
        sigma (float):          Initial estimate of Gaussian width in nm.
        gamma (float):          Initial estimate of Lorentzian width in nm.
        mu (float):             Initial wavelength shift in nm.
        separate_Tvib (bool):   Flag to use different vibrational temperatures for each species
        separate_Trot (bool):   Flag to use different rotational temperatures for each species
        vary_broadening (bool): Flag to optimize broadening parameters during fit, or leave them fixed.
        vary_shift (bool):      Flag to vary the wavelength shift during fitting, or leave it fixed.
        vary_stretch (bool):    Flag to vary the wavelength stretch during fitting, or leave it fixed.
    Returns:
        Parameters:     A lmfit.Parameters instance with parameter values and bounds configured according to the current UI settings.
    """
    #TODO: decide if using `normalize=True` or `False`.
    separate_Tvib = separate_Tvib and (len(species)>1)
    separate_Trot = separate_Trot and (len(species)>1)
    y_min = y.min()
    y_max = y.max()
    y_diff = y_max-y_min
    # var = (y-y_min).std()
    params = lmfit.create_params(**DEFAULT_PARAMS)
    params.add("b", y_min, True, y_min - y_diff/2, y_min + y_diff/2)
    params.add("A", y_diff, vary =True, min = 0, max = y_diff*1.5)
    # params.pop("A")  # use this if using `normalize=False`
    params['sigma'].set(value = sigma, vary=vary_broadening, min=1e-3,max=1)
    params['gamma'].set(value=gamma, vary=vary_broadening, min=1e-3, max=1)
    params['mu'].set(value = mu, vary=vary_shift)
    params['T_rot'].value = Trot
    params['T_vib'].value = Tvib
    weight = 1/len(species)*y_diff
    if separate_Tvib:
        params.pop("T_vib")
    if separate_Trot:
        params.pop("T_rot")
    for specie in species:
        if separate_Trot:
            params.add(f"T_rot_{specie}", value = Trot, **{k:v for k,v in DEFAULT_PARAMS['T_rot'].items() if k in PARAMARGS})
        if separate_Tvib:
            params.add(f"T_vib_{specie}", value = Tvib, **{k:v for k,v in DEFAULT_PARAMS['T_vib'].items() if k in PARAMARGS})
        params.add(f"fraction_{specie}", weight,vary=True,min=0,max=1) # adjust if using `normalize=False`

    
    params.add("lambda_first_order",value=1, vary=vary_stretch, min=0.5, max=2) # scale between half and 2.
    #TODO: Consider if we want to enable second order, or not
    #TODO: are these sensible/workable bounds for second order?
    params.add("lambda_second_order",value=0, vary=vary_stretch, min=-0.5, max=0.5) 
    return params

def scale_wavelengths(x:np.ndarray, first_order=1,second_order=0):
    """Scale the input 1D array of wavelengths by a first and/or second order correction.

    Note:
        This scaling transform applies a stretch/compression to the input wavelengths.
        Scaling is centered on the center of the axis and applied to the relative positions w.r.t. it.
        This should accomodate for non-uniform scaling.

    Args:
        x (np.ndarray): Input array of wavelengths.
        first_order (float): First order correction factor.
        second_order (float): Second order correction factor.

    Returns:
        np.ndarray: Scaled wavelengths.
    """
    if (first_order==1) and (second_order==0):
        return x

    center_index = x.size // 2
    center = x[center_index]

    u = x - center

    scaled = center + first_order * u + second_order * u**2
    return scaled


def molecule_objective(params:"Parameters",x:np.ndarray,y:np.ndarray|None=None,error:np.ndarray|None=None,normalize:bool=False,**dbs) -> "MinimizerResult":
    """Wrapper of the `Moose.lmfit.multi_species_objective` function, with additional wavelength scaling.
    
    It's signature follows the `Moose.lmfit.multi_species_objective` function, and is what is required for use with `lmfit.minimize`.

    That means it needs a `Parameters` object as first argument, with all required parameters defined within it.

    To enable the wavelength scaling feature, the optional parameters `lambda_first_order` and/or `lambda_second_order` need to be included.

    There is fallback handling for when these parameters are not provided, defaulting to no scaling.

    Important:
        When using wavelength scaling, be very carefull about the bounds that are set, as this can significantly affect the fitting results.
        Fitting may become unstable if the scaling parameters are not properly constrained.

    Args:
        params (Parameters): The lmfit Parameters object containing all required parameters. See `Moose.lmfit.multi_species_objective` for details.
        x (np.ndarray): Array of input wavelengths.
        y (np.ndarray|None): Array of observed data corresponding to the wavelengths. Defaults to None.
        error (np.ndarray|None): Array of errors associated with the observed data. Defaults to None.
        normalize (bool): Whether to normalize the objective function. Defaults to False.
        **dbs: any line-by-line database to include in the fit, identified by their key. These keys should be names of parameters in `params` as well, where needed.
    """
    lambda_first_order = float(params.get("lambda_first_order", 1))
    lambda_second_order = float(params.get("lambda_second_order", 0))
    x_scaled = scale_wavelengths(x, first_order=lambda_first_order, second_order=lambda_second_order)
    return Moose.lmfit.multi_species_objective(params, x_scaled, y, error, normalize=normalize, **dbs)