![screenshot](https://github.com/mimurrayy/OES-toolbox/assets/3911345/9eaa9d33-d2ff-423d-a721-5da42fed85d7)

# OES toolbox

Helping out with optical emission spectroscopy of low-temperature plasmas.

- Identifying optical emission lines.
- Fitting molecular spectra.
- Batch processing.
- Free and open-source.

## Citation
Please make sure to cite all appropriate sources when publishing results obtained using the software.
[The wiki has more information.](https://github.com/mimurrayy/OES-toolbox/wiki/How-to-cite)

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


### [**Download**](https://github.com/mimurrayy/OES-toolbox/releases/latest)


Powered by [owl](https://github.com/mimurrayy/owl), [Moose](https://github.com/AntoineTUE/Moose)/[MassiveOES](https://bitbucket.org/OES_muni/massiveoes), [astroquery](https://github.com/astropy/astroquery) and others.   
