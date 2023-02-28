#Cloud Botany material

This repository contains material used to set up and visualize the Cloud Botany LES ensemble experiment.

## Contents

* `ensemble input files/` contains DALES input files for each ensemble member.
Every ensemble member N has a directory `runs/run_N`. The three .nc files in the directory are required for each simulation, and can be copied to every run directory, or symlinked.

* `ensemble generator/` contains Python scripts using EasyVVUQ to sample the parameter space and generate the input files for each ensemble member. 

* 'notebooks/' contains a Jupyter notebook for accessing the Cloud Botany data, producing figures 4 and 5 of the article.

For accessing the Cloud Botany dataset, see [Cloud Botany with DALES](https://howto.eurec4a.eu/botany_dales.html) in the documentation of the EUREC4A intake catalog.


