# Spark

## About

Spark is big data processing engine with machine learning functionality. It's greatest utility is that the parallelization is done automatically. The user writes a script and Spark handles the calculations. It offers multiple frontend languages.

Spark 2.2 is available on the HPC clusters. See the [Python API](https://spark.apache.org/docs/2.2.0/api/python/index.html).

Mllib is the old library. The new one which is based on DataFrames is Spark ML.

[Spark tutorial](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-tut)

[Tuning Spark applications](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-memory)

[Spark application submission via Slurm](https://researchcomputing.princeton.edu/faq/spark-via-slurm)

## Hello World on the HPC Clusters

If you are new to Spark then start by running this simple example:

```bash
$ ssh <YourNetID>@della.princeton.edu
$ cd /scratch/gpfs/<YourNetID>
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop
$ cd hpc_beginning_workshop/RC_example_jobs/spark_big_data
$ wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py
$ sbatch job.slurm
```

## Machine Learning Examples with Spark

The documentaion for the Python API with [Spark ML 2.2](https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html) is here.

## Getting Help

If you encounter any difficulties with getting started on the HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
