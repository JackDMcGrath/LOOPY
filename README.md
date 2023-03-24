# LOOPY - UNDER VERY EARLY STAGE DEVELOPMENT. VERY MUCH USE AT YOUR OWN RISK

LOOPY is an open-source package in Python designed to identify and correct unwrapping errors in interferograms. It is written as an add-on to the COMET [LiCSBAS](https://github.com/yumorishita/LiCSBAS) software.

Having pre-processed their data in LiCSBAS, LOOPY can then be run on the unwrapped interferograms during the LiCSBAS processing chain

LOOPY consists of two steps - mask corrections (LOOPY01_find_errors.py) and linear inversion of loop phase closures (LOOPY03_correction_inversion.py).

To carry out mask correction, LiCSBAS02_ml_prep must have been run. LOOPY01 can be run on this multi-looked data, but it is recommended to use the --fullres flag, as this will mean that errors are detected based on the full resolution tiffs found in the GEOC folder. It can be less effective on multi-looked data, where the sharpness of unwrapping error boundaries can be reduced, and therefore missed. This step works by checking the difference between a pixel and all surrounding pixels - if this change is > 2pi radians, then this is flagged as an error boundary. Regions are flagged as errors if they are isolated from the reference pixel by an error boundary. In this case, we seek to correct this error by comparing the unw values on each side of the error, and trying to reduce it to < 2pi by applying a static offset to the error region. This is particularly effective for catching errors that are isolated from the reference region by full incoherent regions, or areas such as rivers that may have been masked out during unwrapping. If there are known areas that commonly feature an unwrapping error, then these can be ingested and included in the error mapping. Once this correction has been carried out, LiCSBAS may be run normally from step03 onwards as usual.

Linear Inversion of Loop Phase Closures looks to solve loop closures on a pixel by pixel basis, using a linear inversion with L1 Regularisation of the loop phase closures of a small-baseline network, as implemented by MintPy. This is very experimental, and I do not recommend you try using it yet. Currently, it's approach to correcting the network is best approximated thusly https://www.youtube.com/watch?v=m0b_D2JgZgY

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Documentation and Bug Reports

Yet to be written

## Citations


## Acknowledgements

The [Scientific Colour Maps](http://www.fabiocrameri.ch/colourmaps.php) ([Crameri, 2018](https://doi.org/10.5194/gmd-11-2541-2018)) is used in LiCSBAS.

*Jack McGrath\
COMET, School of Earth and Environment, University of Leeds

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/COMET_logo.png"  height="60">](https://comet.nerc.ac.uk/)   [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/logo-leeds.png"  height="60">](https://environment.leeds.ac.uk/see/)  [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/LiCS_logo.jpg"  height="60">](https://comet.nerc.ac.uk/COMET-LiCS-portal/) 
