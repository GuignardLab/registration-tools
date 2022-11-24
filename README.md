# Registration tools

__Before everything else, the current status of the whole thing here is that it only works on UNIX systems (eg Linux & MacOs) that have reasonnable chips (eg not M1 chips for example).__

## Purpose and "history"
This repository is about two scripts to do spatial and temporal registration of 3D microscopy images.
It was initially developed to help friends with their ever moving embryos living under a microscope.
I found that actually quite a few people were interested so I made a  version of it that is somewhat easier to use.

In theory, the main difficulty to make the whole thing work is to install the different libraries.

## Credits
The whole thing is just a wrapping of the amazing blockmatching algorithm developed by [S. Ourselin et al.] and currently maintained Grégoire Malandin et al.@[Team Morphem - inria] (if I am not mistaking).

## Installation

[conda] and [pip] are required to install `registration-tools`

We recommand to install the registration tools in a specific environement (like [conda]). For example the following way:

    conda create -n registration python=3.10
You can then activate the environement the following way:

    conda activate registration

For here onward we assume that you are running the commands from the `registration` [conda] environement.

Then, to install the whole thing, it is necessary to first install blockmatching. To do so you can run the following command:
    
    conda install vt -c morpheme

Then, you can install the 3D-registration library either directly via pip:

    pip install 3D-registration

Or, if you want the latest version, by specifying the git repository:

    pip install git+https://github.com/GuignardLab/registration-tools.git

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
[Team Morphem - inria]: https://team.inria.fr/morpheme/
[conda]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[pip]: https://pypi.org/project/pip/
[manual]: https://github.com/GuignardLab/registration-tools/blob/master/User-manual/user-manual.pdf