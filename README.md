# Covariate-Augmented Probabilistic Topic Models in PySpark

<p align="center">
  <img src="https://dl.dropboxusercontent.com/u/113867121/stmpy/charges.jpg">
</p>

This Python library implements a set of probabilistic topic models that allow integration of arbitrary document-level metainformation into the generative process for the data. The model currently implemented is the [Structural Topic Model](http://structuraltopicmodel.com/) (STM) of Roberts, Stewart, and Tingley. A nonparametric variantis forthcoming. While the software can be run in serial on a local machine, the software's primary goal is use for parallel computation with Apache Spark.
