# EasyVVUQ scripts for generating the Cloud Botany ensemble

This directory contains scripts and settings for
using [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ/)
for generating the ensemble members in the Cloud Botany ensemble.

For seeing the properties of the ensemble members, it's more
convenient to access the parameters in the published dataset (see [How
to EUREC4A](https://howto.eurec4a.eu/botany_dales.html)), or look at
the DALES input files of the members in `ensemble input files/`.

## Requirements

* matplotlib
* numpy
* [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ/) version 1.1.1
* [DALES](https://github.com/dalesteam/dales), for actually running the simulations. We used branch "fugaku".

## Running

```
mkdir workdir
python easyvvuq_dales.py --workdir=workdir  --template=namoptions-botany-fugaku-1536.template --experiment=cloud_botany_7 \
       --prepare --addcenter --runsampler --addpoints 

```

Note: EasyVVUQ seems unable to handle spaces in paths.


Options:

* `--workdir` base directory for EasyVVUQ to use for the model run directories. It creates a subdirectory here for each experiment.
* `--template` a template for the model parameter file, use one of the included files `namoptions*.template`.
* `--experiment` used to select a set of parameters to vary (defined in the easyvvuq_dales.py script). Should match the template.
* `--prepare` the campaign
* `--addcenter` add the center point of the parameter space
* `--runsampler` run the EasyVVUQ sampler. Here it generates the corner points.
* `--addpoints` add points manually defined in the easyvvuq_dales.py script, for the parameter sweeps through the center.


## References

Formulation of the Dutch Atmospheric Large-Eddy Simulation (DALES) and overview of its applications,
T. Heus et al, [Geosci. Model Dev., 3, 415-444, 2010](https://doi.org/10.5194/gmd-3-415-2010)

Assessing uncertainties from physical parameters and modelling choices in an atmospheric large eddy simulation model,
F. Jansson, W. Edeling, J. Attema, and D. Crommelin,
[Philosophical Transactions of the Royal Society A, 379, 20200 073, 2021.](https://doi.org/10.1098/rsta.2020.0073)
The corresponding EasyVVUQ script is here: [EasyVVUQ-DALES](https://github.com/fjansson/EasyVVUQ-DALES).

