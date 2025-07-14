![screenshot](https://github.com/Julian-Held/OES-toolbox/assets/3911345/9eaa9d33-d2ff-423d-a721-5da42fed85d7)

# OES toolbox
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13986864.svg)](https://doi.org/10.5281/zenodo.13986864)



Helping out with optical emission spectroscopy of low-temperature plasmas.

- Identifying optical emission lines.
- Fitting molecular spectra.
- Batch processing.
- Free and open-source.
### [**Download**](https://github.com/Julian-Held/OES-toolbox/releases/latest)

Powered by [owl](https://github.com/Julian-Held/owl), [Moose](https://github.com/AntoineTUE/Moose)/[MassiveOES](https://bitbucket.org/OES_muni/massiveoes), [astroquery](https://github.com/astropy/astroquery) and others.   


## Documentation
* [Installation](https://github.com/Julian-Held/OES-toolbox/wiki/Installation)
* [How to cite](https://github.com/Julian-Held/OES-toolbox/wiki/How-to-cite)
* [Usage](https://github.com/Julian-Held/OES-toolbox/wiki/Usage)

## Supported file types
The software currently supports the following file types:
- Avantes .txt: Supports optional background subtraction and multiple measurements.
- Andor .asc
- Ocean Optics txt
- Other Text/ascii/csv: Automatic determination of headers, delimiters and decimal symbol. Please open an issue with an example if your text file fails to load.
- Avantes .RAW8 (but not .RWD8 or .STR8)
- Andor .sif
- Princeton Instruments .SPE

If you would like support for further file types, please create an issue and provide an example file. The chance of getting support into the software is much higher if you can also provide an example python script on how to load the file (or point me to a python library which can read the file).

## Citation
Julian Held (2024) OES-toolbox: v0.4.0 https://doi.org/10.5281/zenodo.13986864

> [!NOTE]
> Please make sure to cite all appropriate sources when publishing results obtained using the software.
>
> [The wiki has more information.](https://github.com/Julian-Held/OES-toolbox/wiki/How-to-cite)
