# Analysis of live-cell imaging of cohesin

This is a repository to analyze the live-cell imaging data of cohesin and DNA.

# Installation

The conda environment to run the pipeline can be found in the environment folder

To install it type:

```shell
conda env create -f environment/environment_cohesin_analysis.yml

```

# Usage

The workflow is written to be used with [snakemake](https://snakemake.readthedocs.io/en/stable/). No installation is required to run the workflow, simply the conda environment.

The steps and code to execute the workflow is contained in the Snakefile document.

A simplified visual representation of the workflow is shown below:

![pipeline](pipelineAsset_59.png)

To run the pipeline cd in this folder and type:

```shell
snakemake -cores 1

```

With `-cores n`, n being the number of cores you want to use. If no core argument is given, the number of used cores is determined as the number of available CPU cores in the machine.

If you want a dry run showing all the steps

```shell
snakemake -np

```

To visualize the DAG (directed acyclic graph) which corresponds to a vizualization of the workflow with the different steps you can write:

```shell
snakemake --forceall --dag | dot -Tpdf > dag.pdf
```
# Source file

The whole pipeline is written in python, all the function used by snakemake can be found in [utils](utils.py). All the packages necessary for the execution of utils are in the enivironment. In the notebook [test_workflow.ipynb](test_workflow.ipynb) you can run all the steps without using snakemake.

# Notebooks

In this repository you will find jupyter-notebooks allowing to analyze the results of the pipeline. Mainly: [analyze_results.ipynb](analyze_results.ipynb) and [visualize_results.ipynb](visualize_results.ipynb). 