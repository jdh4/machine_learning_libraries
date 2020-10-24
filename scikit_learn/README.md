# Scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) is a Python library for machine learning featuring:

+ Conventional models (not a deep learning library)
+ Comfortable to use (elegant ML pipelines)
+ Interoperable with Pandas
+ Single-node only with multithreading and limited GPU support (XGB)
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

See [this page](https://researchcomputing.princeton.edu/python) for on creating Conda environments for Python packages and writing Slurm scripts.

### Intel

Intel provides their own distribution of Python as well as acceleration libraries for Scikit-learn such as DAAL. You may consider creating your Scikit-learn environment using packages from the `intel` channel:

```
$ module load anaconda3/2020.7
$ conda create --name sklearn-env --channel intel scikit-learn pandas matplotlib
```

## Multithreading

Hyperparameter tuning and cross validation

`n_jobs=-1`

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

module purge
module load anaconda3/2020.7
conda activate sklearn-env

python myscript.py
```

## XGB

## Intel Python Distribution

