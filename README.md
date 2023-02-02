# LOOPY

LOOPY is an open-source package in Python designed to identify and correct unwrapping errors in interferograms. It is written as an add-on to the COMET [LiCSBAS](https://github.com/yumorishita/LiCSBAS) software.

Having pre-processed their data in LiCSBAS, LOOPY can then be run on the unwrapped interferograms. To then include these in further time-series analysis, rerun LiCSBAS from step 12 (Loop Phase Closure Check)

LOOPY consists of two steps - masking unwrapping errors in interferograms, and correcting these errors using a static offset, derived from the loop closures.
Unwrapping errors are detected using an edge detection algorithm, run three times on the modulo 2pi of the original data, the original data + pi, and the original data - pi.
An unwrapping error in this data should appear in the same position in all three iterations.
An region in the interferogram that is isolated from the reference pixel by unwrpping errors are therefore added to a mask.

When fixing unwrapping errors through the loop-closure method, the calculated mask is then applied to any un-corrected interferograms to attempt to prevent the use of 'bad' data in the corrections

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Documentation and Bug Reports

Yet to be written

## Citations


## Acknowledgements

The [Scientific Colour Maps](http://www.fabiocrameri.ch/colourmaps.php) ([Crameri, 2018](https://doi.org/10.5194/gmd-11-2541-2018)) is used in LiCSBAS.

*Jack McGrath\
COMET, School of Earth and Environment, University of Leeds

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/COMET_logo.png"  height="60">](https://comet.nerc.ac.uk/)   [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/logo-leeds.png"  height="60">](https://environment.leeds.ac.uk/see/)  [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/LiCS_logo.jpg"  height="60">](https://comet.nerc.ac.uk/COMET-LiCS-portal/) 
