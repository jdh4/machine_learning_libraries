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

The examples are here:

```bash
/usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/examples/src/main/python/ml
```

```bash
#!/bin/bash
#SBATCH --job-name=spark-ml      # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=3      # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G                # memory per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2019.10
module load spark/hadoop2.7/2.2.0

spark-start
spark-submit --total-executor-cores 24 --executor-memory 4G pi.py 100
/usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/examples/src/main/python/ml/als_example.py
```

### 


## Getting Help

If you encounter any difficulties with using Spark on the HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a walk-in <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
