# Analysis of live-cell imaging of cohesin

This is a repository to analyze the live-cell imaging data of cohesin and DNA.

# Installation

The conda environment to run the pipeline can be found in the environment folder

To install it type

```
conda env create -f environment/environment_cohesin_analysis.yml

```

# Usage

The workflow is written to be used with [snakemake](https://snakemake.readthedocs.io/en/stable/). No installation is required to run the workflo, simply the conda environment.

The steps and code to execute the workflow is contained in the Snakefile document. All the function that are ran in the snakefile can be found in [utils](utils.py). 

To run the pipeline cd in this folder and type:

```
snakemake -cores 1

```

With -cores being the number of cores you want to use.

If you want a dry run showing all the steps

```
snakemake -np

```