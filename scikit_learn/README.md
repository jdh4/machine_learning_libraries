# Scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) is a Python library for machine learning featuring:

+ Conventional models (not a deep learning library)
+ Comfortable to use (elegant ML pipelines)
+ Interoperable with Pandas
+ Single-node only with multithreading and limited GPU support (XGB)
+ Excellent [documentation](https://scikit-learn.org/stable/user_guide.html) (good way to prepare for interviews (ML engineer, data scientist)

For an introduction to machine learning and Scikit-learn see this GitHub [repo and book](https://github.com/ageron/handson-ml2) by Aurelien Geron.

## Installation

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
$ conda create --name sklearn-env --channel <some-channel> scikit-learn matplotlib <another-package>
```

See [this page](https://researchcomputing.princeton.edu/python) for on creating Conda environments for Python packages and writing Slurm scripts.

## Multithreading

Hyperparameter tuning and cross validation

`n_jobs=-1`

## XGB

## Intel Python Distribution

