# Registration tools

## Purpose and "history"
This repository is about two scripts to do spatial and temporal registration of 3D microscopy images.
It was initially developed to help friends with their ever moving embryos living under a microscope.
I found that actually quite a few people were interested so I made a  version of it that is somewhat easier to use.

In theory, the main difficulty to make the whole thing work is to install the different libraries.

## Credits
The whole thing is just a wrapping of the amazing blockmatching algorithm developed by [S. Ourselin et al.] and currently maintained Grégoire Malandin et al.@[Team Morpheme - inria] (if I am not mistaking).

## Installation

[conda] and [pip] are required to install `registration-tools`

We recommand to install the registration tools in a specific environement (like [conda]). For example the following way:

    conda create -n registration python=3.10
You can then activate the environement the following way:

    conda activate registration

For here onward we assume that you are running the commands from the `registration` [conda] environement.

Then, to install the whole thing, it is necessary to first install blockmatching. To do so you can run the following command:
    
    conda install vt -c morpheme

or,

    conda install vt-python -c morpheme -c conda-forge


Then, you can install the 3D-registration library either directly via pip:

    pip install 3D-registration

Or, if you want the latest version, by specifying the git repository:

    pip install git+https://github.com/GuignardLab/registration-tools.git

### Troubleshooting
- Windows:

    If you are trying to run the script on Windows you might need to install `pthreadvse2.dll`. 

    It can be found there: https://www.pconlife.com/viewfileinfo/pthreadvse2-dll/ . Make sure to download the version that matches your operating system (32 or 64 bits, most likely 64).

## Usage

Most of the description on how to use the two scripts is described in the [manual] (Note that the installation part is quite outdated, the remaining is ok).

That being said, once installed, one can run either of the scripts from anywhere in a terminal by typing:

    time-registration

or

    spatial-registration

The location of the json files or folder containing the json files will be prompted and when provided the registration will start.

It is also possible to run the registration from a script/notebook the following way:
```python
from registrationtools import TimeRegistration
tr = TimeRegistration('path/to/param.json')
tr.run_trsf()
```

or

```python
from registrationtools import TimeRegistration
tr = TimeRegistration('path/to/folder/with/jsonfiles/')
tr.run_trsf()
```

or

```python
from registrationtools import TimeRegistration
tr = TimeRegistration()
tr.run_trsf()
```

and a path will be asked to be inputed.

### Example json files

Few example json files are provided to help the potential users. You can find informations about what they do in the [manual].

[S. Ourselin et al.]: http://www-sop.inria.fr/asclepios/Publications/Gregoire.Malandain/ourselin-miccai-2000.pdf
[Team Morpheme - inria]: https://team.inria.fr/morpheme/
[conda]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[pip]: https://pypi.org/project/pip/
[manual]: https://github.com/GuignardLab/registration-tools/blob/master/User-manual/usage/user-manual.pdf
