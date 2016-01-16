#!/bin/bash
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/00-pyspark-setup.py
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/jupyter_notebook_config.py
wget https://dl.dropboxusercontent.com/u/113867121/stmpy/kernel.json
sudo cat 00-pyspark-setup.py >> ~/.ipython/profile_pyspark/startup/00-pyspark-setup.py
sudo mkdir -p ~/.ipython/kernels/pyspark
sudo cat kernel.json >> ~/.ipython/kernels/pyspark/kernel.json
jupyter notebook --generate-config
sudo rm ~/.jupyter/jupyter_notebook_config.py
sudo cat jupyter_notebook_config.py >> ~/.jupyter/jupyter_notebook_config.py
