#!/bin/bash
sudo yum -y update
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/stmpy-1.0.0.tar.gz
sudo yum -y install git
echo -e "\nalias python='python27'" >> ~/.bash_profile
source ~/.bash_profile
sudo yum -y install readline-devel
sudo pip-2.7 install rpy2
sudo pip-2.7 install pandas
sudo pip-2.7 install stmpy-1.0.0.tar.gz
echo -e "packages <- c('Matrix', 'stringr', 'splines', 'matrixStats', 
	'slam', 'lda', 'glmnet', 'magrittr', 'tm', 'SnowballC') \n

    if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
        install.packages(setdiff(packages, rownames(installed.packages())), 
            repos='http://cran.us.r-project.org')}" >> setup.R
sudo Rscript setup.R
sudo pip-2.7 install ipython notebook findspark

ipython profile create pyspark
export SPARK_HOME=/usr/lib/spark
echo -e "import os
import sys

spark_home = os.environ.get('SPARK_HOME', None)
if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))

execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9999" >> ~/.ipython/profile_pyspark/startup/00-pyspark-setup.py

mkdir -p ~/.ipython/kernels/pyspark
echo -e '{"display_name": "pySpark (Spark 1.4.0)","language": "python","argv": [ "/usr/bin/python27", "-m", "IPython.kernel", "--profile=pyspark", "-f", "{connection_file}"]}' >> ~/.ipython/kernels/pyspark/kernel.json

jupyter notebook --generate-config
sudo rm ~/.jupyter/jupyter_notebook_config.py
echo -e "c = get_config()
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
" >> ~/.jupyter/jupyter_notebook_config.py
