# Code for anomaly detection and clustering of energy time series (Swisscom/Migros data)

This repository contains multiple studies done within in the framework of the Anomaly detection project in coordination with Swisscom/OPIT. These studies are essentially represented by a few (python) jupyter notebooks, in which all analyses were conducted. Note that we had access to Swisscom and Migros datasets, which had very different patterns and therefore were treated in separated studies.

## Contents of the notebooks:

### Notebooks on Migros data:

1. "Migros_data_tests" notebook: first analysis on migros data, plot images to see how stable the data is…
2. "Migros_CLUSTERING" notebook: clustering tests of buildings on the migros data (Fourier features clustering with DBSCAN)
3. "Migros_CLUSTERING_weeks" notebook: clustering tests of 'weeks' on the migros data (Fourier features clustering with DBSCAN)

Note that these notebooks call sources of data, which need to be downloaded from the ENAC sharing platform: https://enacshare.epfl.ch/drHapXYmEvVqDibyL2ezu
and need to be put in the /data folder.

Note also that "TrainingMethods188.py" is a python script containing classes and functions to ease the training of SVM and RFs. Other functions used are contained in the helpers_* files.
