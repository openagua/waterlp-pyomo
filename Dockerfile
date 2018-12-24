FROM ubuntu:latest
MAINTAINER David Rheinheimer "drheinheimer@umass.edu"

ARG VERSION=0.1

RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y glpk-utils
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip

COPY requirements.txt /home/requirements.txt
ADD /waterlp /home/waterlp

WORKDIR /home
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /waterlp
#RUN python3 ./setup.py build_ext --inplace
