# Overview

This is a demand-driven, priority-based model developed for OpenAgua, similar in concept to [WEAP](http://weap21.org/). Key differences include:
* demand and priority can both be specified as arrays ("blocks"), to allow for piecewise linear changes in water value
* within each block, demand/priority arrays can be automatically divided into equal parts ("subblocks") with quadratic value functions to allocated shortages in a second level of piecewise linearization

Some additional key points:
* Currently uses [Pyomo](http://www.pyomo.org/) for model formulation
* Uses [GLPK](https://www.gnu.org/software/glpk/) for the LP solver

# Installation

## Direct installation
Direct installation is for those who want to run a Python script directly, including for development. The general process is:

1. Install Python 3.6, ideally within a new environment:
* Install [Python 3.6](https://www.python.org/downloads/release/python-366/)
* Install software needed to compile some of the Python packages.
* Create a virtual environment with Python 3.6. Follow [these instructions](https://medium.com/@peterchang_82818/python-beginner-must-know-virtualenv-tutorial-example-5e3f82cfbd8b). **NOTE**: Virtual environments are an annoying part of Python, especially if you are coming from, say, R or Matlab. They aren't strictly required, but are generally recommended to ensure correct versions of libraries are installed for each project. Most of the top Python IDEs (e.g., PyCharm, Wing IDE, Spyder, etc.) have tools built in to help create and manage virtual environments. PyCharm is particularly good.
1. Clone this repository into a folder of your choice.
1. Install (from within your virtual environment, if used) required packages with pip: `pip install -r requirements.txt`.


## Docker installation

Docker is a system for packaging and deploying software in a way that ensures cross-platform consistency.

1. Install [Docker](https://www.docker.com/get-started)
1. Install the Docker image, available as [openagua/waterlp-general-pyomo](https://hub.docker.com/r/openagua/waterlp-general-pyomo/)

# Instructions for use
(documentation forthcoming)

# Troubleshooting
(documentation forthcoming)
