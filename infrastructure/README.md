# Project Setup
This page guides you through the setup of all necessary tools which are used for data processing and analysis in this project.

## Mambaforge
Our recommended tool for environment managemt is [Mambaforge](https://github.com/conda-forge/miniforge).
Please download the latest version and install it if you have not done so already.

### Build environment
An environment is a collection of tools, packages and dependencies which are required to execute a script.
With mambaforge we have an environment manager which we can use to build envrionments by providing an environment recipe stored as `.yaml` file.

```{attention}
Make sure that you are in the root-directory of your project.
```

With the following command a new envrionment is created with the name `cohesin_analysis`.
The packages which are installed are defined in the file `infrastructure/env-yamls/environment_cohesin_analysis.yml`.

```shell
conda env create -f infrastructure/env-yamls/environment_cohesin_analysis.yml

```

### Activate environment
Once an envrionment is created it must be activated such that we can use the installed packages.

```bash
mamba activate cohesin_analysis
```

Once the environment is activate your CLI prompt will be prefixed with the environment name e.g.:
`(cohesin_analysis)$ `

