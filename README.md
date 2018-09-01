This is a priority-based model developed for OpenAgua. Some general points, pending full documentation:

Methods:
* Demand-driven, priority-based, LP-based similar to [WEAP](http://www.weap21.org)
* Demand & priorities can be piecewise-linear
* Currently uses [Pyomo](http://www.pyomo.org/) for model formulation
* Uses [GLPK](https://www.gnu.org/software/glpk/) for the LP solver

Use:
* Via code directly
* Via Docker: [openagua/waterlp-general-pyomo](https://hub.docker.com/r/openagua/waterlp-general-pyomo/)
* Via OpenAgua

Future documentation to include:
* Installation
* Use
* Troubleshooting
* Development
