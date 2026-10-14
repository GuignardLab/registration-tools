# Registrationtools docker

The dockerfile makes an image that contains a set of relevant packages for image registration in python.

The relevant packages for image registration that contains are:

 - [`registrationtools`](https://github.com/GuignardLab/registration-tools): This package
 - [`vt-python`](https://gitlab.inria.fr/morpheme/vt-python): Original package over which this package makes a wrapper. 
 - [`SimpelITK`](https://simpleitk.org/): Classical package for image registration, wrapper over the ITK package.
 - [`skimage`](https://scikit-image.org/): Well-stablished package for image manipulation in python based on numpy.

In addition to these packages, other packages are installed (e.g. numpy, matplotlib...). A list can be seen in environment.yaml.

Constructed images are already deposited in [dsblab/registrationtools](https://hub.docker.com/repository/docker/dsblab/registrationtools), so there is not need of building the image except if you want to extend further capabilities.

# Running the image

## Interactive Shell

Start the container in interactive format.

```shell
docker run -it \
    --mount type=bind,source="$(pwd)",target=/home \
    dsblab/registrationtools:v0.1
```

Activate the conda image.

```shell
source activate registration
cd home
```

And now all the packages and modules will be loaded. You can start a python shell:

```shell
python
```

or execute packages installed from the command line (e.g. blockmatching)

```shell
blockmatching -h
```

## Script

To execute directly a bash script simply

```shell
docker run \
    --mount type=bind,source="$(pwd)",target=/home \
    dsblab/registrationtools:v0.1 /bin/bash -c "source activate registration: cd home; <bash_script.sh>"
```

and a python script

```shell
docker run \
    --mount type=bind,source="$(pwd)",target=/home \
    dsblab/registrationtools:v0.1 /bin/bash -c "source activate registration; cd home; python <python_script.py>"
```

## Jupyter lab

If you want to work interactively with a jupyter notebook.

```shell
docker run -it \
    -p 8888:8888 \
    --mount type=bind,source="$(pwd)",target=/home \
    dsblab/registrationtools:v0.1 /bin/bash -c "jupyter lab --notebook-dir=/home --ip='*' --port=8888 --allow-root"
```

You can then view the Jupyter Notebook by opening http://localhost:8888 in your browser, or http://<DOCKER-MACHINE-IP>:8888 if you are using a Docker.