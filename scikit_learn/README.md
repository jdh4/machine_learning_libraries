# Scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) is a Python library for machine learning featuring:

+ Conventional models (not a deep learning library)
+ Comfortable to use (elegant ML pipelines)
+ Interoperable with Pandas
+ Single-node only with multithreading, very limited GPU support (XGB)
+ Excellent [documentation](https://scikit-learn.org/stable/user_guide.html) (good way to prepare for ML engineer and data scientist interviews)

For an introduction to machine learning and Scikit-learn see this GitHub [repo and book](https://github.com/ageron/handson-ml2) by Aurelien Geron.

## Installation

### Anaconda

Scikit-learn is pre-installed as part of the Anaconda Python disribution:

```
$ module load anaconda3/2020.7
(base) $ python
>>> import sklearn
>>> sklearn.__version__
'0.23.1'
```

If you need additional packages that are not found in the Anaconda distribution then make your own Conda environment:

```
$ module load anaconda3/2020.7
$ conda create --name sklearn-env --channel <some-channel> scikit-learn pandas matplotlib <another-package>
```

See [this page](https://researchcomputing.princeton.edu/python) for more on creating Conda environments for Python packages and writing Slurm scripts.

### Intel

Intel provides their own distribution of Python as well as acceleration libraries for Scikit-learn such as [DAAL](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html). You may consider creating your Scikit-learn environment using packages from the `intel` channel:

```
$ module load anaconda3/2020.7
$ conda create --name sklearn-env --channel intel scikit-learn pandas matplotlib
```

## Multithreading

Scikit-learn depends on the `intel-openmp` package which enabling multithreading. This allows the software to use multiple CPU-cores for hyperparameter tuning, cross validation and other embarrassingly parallel operations. If you are calling a routine that takes the `n_jobs` parameter then set this to `n_jobs=-1` to take advantage of all the CPU-cores in your Slurm allocation.

Below is an appropriate Slurm script for a Scikit-learn job that takes advantage of multithreading:

```
#!/bin/bash
#SBATCH --job-name=sklearn       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=<YourNetID>@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.7
conda activate sklearn-env

python myscript.py
```

## Example Job


## Another Example

Take a look at an end-to-end ML project [here](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb) for predicting housing prices in California.


## Gradient Boosting Models and GPUs

[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)

[XGBoost](https://xgboost.readthedocs.io/en/latest/)

