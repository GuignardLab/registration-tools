# Registration tools

__Before everything else, the current status of the whole thing here is that it only works on UNIX systems (eg Linux & MacOs) that have reasonnable chips (eg not M1 chips for exmaple).__

This repository is about two scripts to do spatial and temporal registration of 3D biological images. It was initially developed to help friends with there ever moving embryos living under a microscope. I found that actually quite a few people were interested so I made a somewhat easier to use version of it.

The whole thing is just a specific wrapping of the amazing blockmatching algorithm developed by [S. Ourselin et al.] and currently maintained Grégoire Malandin et al.@[Team Morphem - inria] (if I am not mistaking).

## Installation

[conda] and [pip] are required to install the regisrtation-tools

We recommand to install the registration tools in a separate environement (like [conda]). For example the following way:

    conda create -n registration python=3.10

Then, to install the whole thing, it is necessary to first install blockmatching. To do so you can run the following command:
    
    conda install vt -c morpheme

Then, you can install the current library by either cloning/downloading the repository and installing it from there. For example the following way:

    git clone https://github.com/leoguignard/registration-tools.git
    cd registration-tools
    pip install .

## Usage

Most of the description on how to use the two scripts is described in the [manual].

That being said, once installed, one can run either of the scripts from anywhere in a terminal by typing:

    time-regitration.py

or

    spatial-registration.py

The location of the json files or folder containing the json files will be prompted and when provided the registration will start.

[S. Ourselin et al.]: http://www-sop.inria.fr/asclepios/Publications/Gregoire.Malandain/ourselin-miccai-2000.pdf
<!-- [Grégoire Malandin et al.]: https://gitlab.inria.fr/morpheme/vt -->
<!-- [morphem gitlab]: https://gitlab.inria.fr/morpheme/vt -->
[Team Morphem - inria]: https://team.inria.fr/morpheme/
[conda]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[pip]: https://pypi.org/project/pip/
[manual]: https://github.com/leoguignard/registration-tools/blob/master/User-manual/user-manual.pdf