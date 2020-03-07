# Spark

Spark is big data processing engine with modules for streaming, SQL, graphs, machine learning and more. Its greatest utility is that the parallelization and processing of the data is automatically handled. The are four different Spark API languages: Scala, Python, Java and R.

- The basic idea is to store the data in a DataFrame which is distributed over many nodes. The underlying data structure is the resilent distributed datasets (RDD).  Think of an RDD of a list of objects.

- Lazy evaluation is used. Operations in the Spark script which transform an RDD are translated to a node in the computation graph. Actions cause the graph to be evaluated. Intermediate results can be cached in memory and/or disk.

Spark 2.2 is available on the Princeton HPC clusters. See the [Python API](https://spark.apache.org/docs/2.2.0/api/python/index.html).

## A Simple DataFrame

The session below illustrates how to create a simple DataFrame in the PySpark shell during an interactve session:

```bash
$ ssh della  # or another cluster
$ salloc --nodes=1 --ntasks=1 --time=10
$ module load anaconda3 spark/hadoop2.7/2.2.0
$ spark-start
$ pyspark

>>> myRDD = sc.parallelize([('Denver', 5280), ('Albuquerque', 5312), ('Mexico City', 7382)])
>>> df = sqlContext.createDataFrame(myRDD, ['City', 'Elevation'])
>>> df.show()
+-----------+---------+
|       City|Elevation|
+-----------+---------+
|     Denver|     5280|
|Albuquerque|     5312|
|Mexico City|     7382|
+-----------+---------+
>>> df = df.filter(df["Elevation"] < 6000)
>>> df.show()
+-----------+---------+
|       City|Elevation|
+-----------+---------+
|     Denver|     5280|
|Albuquerque|     5312|
+-----------+---------+
>>> exit()
$ exit
```

## Hello World with the Slurm Job Scheduler

If you are new to Spark then start by running this simple example:

```bash
$ ssh <YourNetID>@della.princeton.edu
$ cd /scratch/gpfs/<YourNetID>
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop
$ cd hpc_beginning_workshop/RC_example_jobs/spark_big_data
$ wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py
$ sbatch job.slurm
```

## Machine Learning

Spark ML is the machine learning component of Spark. The previous library was called mllib.

The documentaion for the Python API with [Spark ML 2.2](https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html) is here.

The Spark 2.2 machine learning examples are here:

```bash
# ssh tiger, della, perseus, adroit
$ cd /usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/examples/src/main
$ ls
java  python  r  resources  scala
$ cd /usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/examples/src/main/python/ml
$ ls
aft_survival_regression.py                   logistic_regression_with_elastic_net.py
als_example.py                               max_abs_scaler_example.py
binarizer_example.py                         min_hash_lsh_example.py
bisecting_k_means_example.py                 min_max_scaler_example.py
bucketed_random_projection_lsh_example.py    multiclass_logistic_regression_with_elastic_net.py
bucketizer_example.py                        multilayer_perceptron_classification.py
chisq_selector_example.py                    naive_bayes_example.py
chi_square_test_example.py                   n_gram_example.py
correlation_example.py                       normalizer_example.py
count_vectorizer_example.py                  onehot_encoder_example.py
cross_validator.py                           one_vs_rest_example.py
dataframe_example.py                         pca_example.py
dct_example.py                               pipeline_example.py
decision_tree_classification_example.py      polynomial_expansion_example.py
decision_tree_regression_example.py          quantile_discretizer_example.py
elementwise_product_example.py               random_forest_classifier_example.py
estimator_transformer_param_example.py       random_forest_regressor_example.py
fpgrowth_example.py                          rformula_example.py
gaussian_mixture_example.py                  sql_transformer.py
generalized_linear_regression_example.py     standard_scaler_example.py
gradient_boosted_tree_classifier_example.py  stopwords_remover_example.py
gradient_boosted_tree_regressor_example.py   string_indexer_example.py
imputer_example.py                           tf_idf_example.py
index_to_string_example.py                   tokenizer_example.py
isotonic_regression_example.py               train_validation_split.py
kmeans_example.py                            vector_assembler_example.py
lda_example.py                               vector_indexer_example.py
linear_regression_with_elastic_net.py        vector_slicer_example.py
linearsvc.py                                 word2vec_example.py
logistic_regression_summary_example.py
```

Below is `random_forest_classifier_example.py`:

```python
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only
    # $example off$

    spark.stop()
```

Run the commands below:

```bash
$ git clone
$ cd intro_ml_libs/spark
$ cp /usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/examples/src/main/python/ml/random_forest_classifier_example.py .
$ cp /usr/licensed/spark/spark-2.2.0-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt .
$ sbatch job.submit
```

You can see the updated examples on [GitHub](https://github.com/apache/spark/tree/master/examples/src/main).

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

### Spark at Princeton

[Spark tutorial](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-tut)

[Tuning Spark applications](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-memory)

[Spark application submission via Slurm](https://researchcomputing.princeton.edu/faq/spark-via-slurm)


## Getting Help

If you encounter any difficulties with using Spark on the HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a walk-in <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
