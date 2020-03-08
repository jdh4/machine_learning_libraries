# R

R is programming language popular in the statistical sciences. There are 19000 packages on CRAN.

For using R on the HPC clusters: [https://researchcomputing.princeton.edu/R](https://researchcomputing.princeton.edu/R)

See [this page](https://cran.r-project.org/web/views/MachineLearning.html) for a list of ML packages on CRAN.

## Caret

Install the needed packages:

```
$ ssh adroit
$ module load rh/devtoolset/8
$ R
> install.packages("caret")
> q()
```

Below is the R script:

```R
library(caret)
```

Now submit the job:

```
$ cd intro_ml_libs/R/myjob
$ wget data
$ sbatch job.slurm
```

The output should be:

## R and Deep Learning

[MXNet](https://mxnet.apache.org/api/r)
 >  MXNet supports the R programming language. The MXNet R package brings flexible and efficient GPU computing and state-of-art deep learning to R. It enables you to write seamless tensor/matrix computation with multiple GPUs in R. It also lets you construct and customize the state-of-art deep learning models in R, and apply them to tasks, such as image classification and data science challenges.


[R interface to keras](https://www.amazon.com/Deep-Learning-R-Francois-Chollet/dp/161729554X/ref=sr_1_3?keywords=deep+learning+with+R&qid=1583689546&sr=8-3)
