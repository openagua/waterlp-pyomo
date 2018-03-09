FROM ubuntu:latest
MAINTAINER David Rheinheimer "drheinheimer@umass.edu"

ARG VERSION=0.1

RUN apt-get update && apt-get install -y \
    build-essential \
    glpk-utils \
    python3 \
    python3-pip

WORKDIR ~

ADD /model /model
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /model
RUN python3 ./setup.py build_ext --inplace
