# 21F2000709_MLOPS_WEEKLY_ASSIGNMENT

Week 3 Assignment

Objective:
Integrating Feast Feature Store into the IRIS Pipeline using GCS

Utility of each file:
- Feast_Assignment.ipynb - It is the jupyter notebook file containing the whole code which is responsible for 
    - Initialize the Feast Feature Repository
    - Generating configuration file for Feast Feature Store
    - Downloading iris.csv from GCP Bucket
    - Modifying the data to make it compatible for feast feature store
    - Defining Entities, Data Sources & Feature Views
    - Applying Definitions & Materializing Features
    - Fetch Features for Training
    - Training the model
    - Fetch Features for Inference
    - Doing the Inference
    - Outputs of everything that is run


- feast_iris/feature_repo/feature_store.yaml - It is the configuration file for the Feast Feature Store generated during the running of the "Feast_Assignment.ipynb".

- feast_iris/feature_repo/iris_features.py - It is the feature definition file for the Feast Feature Store generated during the running of the "Feast_Assignment.ipynb". It contains definitions of File Source, Entities, FeatureView for the Feast Feature Store.
