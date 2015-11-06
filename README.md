# Covariate-Augmented Probabilistic Topic Models in PySpark

*This repository is under active development.*

This Python library implements a set of probabilistic topic models that allow integration of arbitrary document-level metainformation into the generative process for the data. The models implemented include the [Structural Topic Model](http://structuraltopicmodel.com/) (STM) of Roberts, Stewart, and Tingley, as well a nonparametric variant, Covariate-Augmented Nonparametric Latent Dirichlet Allocation (C-LDA). While the software can be run in serial on a local machine, all the methods support parallel computation using Apache Spark. The package includes tools for model selection, visualization, and estimation of topic-covariate regressions.
