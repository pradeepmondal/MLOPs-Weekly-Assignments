# 21F2000709_MLOPS_WEEKLY_ASSIGNMENT

Week 1 Assignment

Objective:
Setting up the ML pipeline for IRIS Classifier in Vertex AI platform using GCS

Utility of each file:
- training_notebook.ipynb - It has the code for setting up the Vertex AI sdk, cloud storage bucket, fetching the dataset, creating splits, training decision tree model, storing the model artifact and log_text in timestamped folder in the storage bucket.

- inference_notebook.ipynb - This script loads the eval set and trained model from the gc bucket, do inference and store the results in timestamped folders in inference_runs folder.

- outputs [Folder] - contains the output log files of different training and inference runs.