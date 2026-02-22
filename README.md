# 21F2000709_MLOPS_WEEKLY_ASSIGNMENT

Week 2 Assignment

Objective:
Incorporating Data Version Control (DVC) into the IRIS machine learning pipeline Classifier using GCS

Utility of each file:
- code/train.py - It has the code for training decision tree model using data/data.csv (to be downloaded from GCS Bucket) and saving the model.

- code/requirements.txt - It has the required dependencies(including dvc) for running train.py and also for data versioning.

- shell_session.txt - contains the gcp shell session interaction. In lines 615 - 688, it has the verification that the local files revert correctly to the selected version, then return to the latest version.