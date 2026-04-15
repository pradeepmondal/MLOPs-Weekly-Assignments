Model Card: IRIS Decision Tree Classifier
1. Model Details
Architecture: Scikit-Learn DecisionTreeClassifier

Hyperparameters: max_depth = 1

Frameworks: Scikit-Learn, Fairlearn, SHAP, Evidently AI

Artifact: iris_model.pkl

2. Intended Use
Intended Users: Data science students, MLOps engineers, and researchers.

3. Training Data Description
Source: Classic Iris dataset, hosted in Google Cloud Storage (gs://mlops-assignment-mlops-iitmadras-trialrun/data/raw/iris.csv).

Features Used: sepal_length, sepal_width, petal_length, petal_width.

Target Variable: species (setosa, versicolor, virginica).

Data Split: 150 total samples (105 Training samples, 45 Testing samples).

Sensitive Attribute: A synthetic location feature (values 0 or 1) was randomly generated and assigned to each sample. This attribute was dropped from the training features to prevent the model from learning from it, reserving it purely for the post-training fairness audit.

4. Performance Metrics
The model is intentionally underfit to demonstrate clear explainability rules and the impact of single-feature reliance.

Overall Accuracy: 71.11% (0.7111)

Fairness Assessment (Disaggregated by Location)
Performance was audited across the synthetic location groups using Fairlearn.

Location 0: * Accuracy: 66.67%

Precision: 46.67%

Recall: 66.67%

Location 1: * Accuracy: 74.07%

Precision: 52.08%

Recall: 66.67%

Fairness Observation: There is a ~7.4% accuracy gap between Location 0 and Location 1. Because the locations were randomly assigned, this disparity is entirely due to the statistical noise of a small test set. However, it successfully demonstrates the pipeline's ability to detect performance gaps across demographic groups.

5. Model Explainability
Audited using a SHAP TreeExplainer, refer to main.ipynb for shap summary plots.

6. Known Limitations & Risks
Sample Size Limit: Trained on a historical, narrow dataset of 150 samples, making it unsuitable for generalizing to unseen sub-species or regional variations.