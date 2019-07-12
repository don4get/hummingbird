# Hummingbird [![Build Status](https://travis-ci.org/don4get/hummingbird.svg?branch=master)](https://travis-ci.org/don4get/hummingbird)
![](hummingbird/resources/hummingbird.gif)
### Incentive

**hummingbird** aims to model fixedwind UAVs and allow their sizing / tuning.

### Credits

The project is initially forked from [mavsim_template_files](https://github.com/sethmnielsen/mavsim_template_files), 
completed by [sethmnielsen](https://github.com/sethmnielsen) and 
created by [Pr. R. Beard](https://github.com/randybeard) & Pr. T. McLain.
These previous templates are the counterpart of the book 
[Small UAV: Theory and Practice](https://press.princeton.edu/titles/9632.html) 
written by the same two professors.

I already completed the suggested template project in Matlab and in python (see [limbo](https://github.com/don4get/limbo)).
This new version will allow me to go further and test some features of OSS scientific libraries I am involved into 
([python-control](https://github.com/python-control/python-control), [sippy](https://github.com/CPCLAB-UNIPI/SIPPY)).

### Setup
* Install recent python 3.7
* `pip install -r requirements.txt`
* `python ./chap12/mavsim_chap12.py`