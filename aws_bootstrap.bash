#!/bin/bash
sudo yum -y update
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/stmpy-1.0.0.tar.gz
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/00-pyspark-setup.py
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/jupyter_notebook_config.py
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/kernel.json
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