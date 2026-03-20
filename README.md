# 21F2000709_MLOPS_WEEKLY_ASSIGNMENT

Week 5 Assignment

Objective:
Integrating MLFlow into the IRIS Pipeline

Utility of each file:
- tests/test_data.py - tests for validating the datasets
- tests/test_model.py - tests for model evaluation
- train.py - python file containing the code for training the DecisionTreeClassifier model as well as integration of mlflow for logging and registering models
- requirements.txt - contains the dependencies required for running train.py and tests
- .github/workflows/ci.yaml - contains the code for CI for every push and PR (for github actions)
- mlflow.db - contains the structured tracking metadata for experiment runs, etc.
- *.dvc (files) - pointers to the actual files (used for versioning in DVC)
- .dvc [Folder] - contains required files including configuartion files required by DVC
- .dvcignore - tells DVC which files/folder to ignore
- .gitignore - tells Git which files/folder to ignore
