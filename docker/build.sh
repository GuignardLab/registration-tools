#!/bin/bash

version=v0.1

docker build . -t dsblab/registrationtools:$version
docker push dsblab/registrationtools:$version