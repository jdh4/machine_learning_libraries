# Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/) is a machine learning library:

+ Developed by IBM
+ Distributed training
+ GPU acceleration and supports sparse data structures
+ `sklearn` interface
+ `snapml-spark` offers distributed training and integrates with PySpark but only offers three models (linear regression, logistic regression, linear support vector classifier (SVC)

See the [documentation](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/index.html) (v1.6.0) and [tutorials](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/tutorials.html).

## Installation

### Traverse

```
$ conda create --name pai4sk-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda pai4sk scikit-learn
```

```
$ module load anaconda3/2020.7
$ conda create --name snap-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access snapml
```

### TigerGPU, Adroit

Only the Spark interface is available for the `x86_64` architecture:

```
$ module load anaconda3/2020.7
$ conda create --name snap-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda snapml-spark
```

## Example Job

```python
from pai4sk import RandomForestClassifier as SnapForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

rf = SnapForest(n_estimators=50, n_jobs=4, max_depth=8, use_histograms=True, use_gpu=True, gpu_ids=[0])
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy = {100 * acc:.1f}%")
```

The output of the code is:

```
Accuracy = 95.6%
```

Below is the corresponding Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node

module purge
module load anaconda3/2020.7
conda activate pai4sk-env

python myscript.py
```

## Notes

When trying to do the `iris` dataset:

```
ValueError: Multiclass classification not supported for decision tree classifiers.
```
