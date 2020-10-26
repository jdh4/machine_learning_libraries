# Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/) is a machine learning library:

+ Developed by IBM
+ Distributed training
+ GPU acceleration and supports sparse data structures
+ `sklearn` interface
+ `snapml-spark` offers distributed training and integrates with PySpark using a `spark.ml` interface

See the [documentation](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/index.html) (v1.6.0) and [tutorials](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/tutorials.html).

## Installation

### Traverse

```
$ conda create --name pai4sk-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda pai4sk
```

```
$ module load anaconda3/2020.7
$ conda create --name snap-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access snapml
```

### TigerGPU, Adroit

```
$ module load anaconda3/2020.7
$ conda create --name snap-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda snapml-spark
```

## Example Job

