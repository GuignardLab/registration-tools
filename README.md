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

Then, you can install the current library by either cloning/downloading the repository and installing it from there. For example the following way:

    git clone https://github.com/leoguignard/registration-tools.git
    cd registration-tools
    pip install .

## Usage

Most of the description on how to use the two scripts is described in the [manual] (Note that the installation part is quite outdated, the remaining is ok).

That being said, once installed, one can run either of the scripts from anywhere in a terminal by typing:

    time-regitration.py

or

    spatial-registration.py

The location of the json files or folder containing the json files will be prompted and when provided the registration will start.

### Example json files

Few example json files are provided to help the potential users. You can find informations about what they do in the [manual].

[S. Ourselin et al.]: http://www-sop.inria.fr/asclepios/Publications/Gregoire.Malandain/ourselin-miccai-2000.pdf
[Team Morphem - inria]: https://team.inria.fr/morpheme/
[conda]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[pip]: https://pypi.org/project/pip/
[manual]: https://github.com/leoguignard/registration-tools/blob/master/User-manual/user-manual.pdf