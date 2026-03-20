import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

GCS_ARTIFACT_PATH = "gs://week-5-mainrun-mlops-iitmadras/mlflow-artifacts"

experiment_name = "Iris_DecisionTree_Hyperopt"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name, artifact_location=GCS_ARTIFACT_PATH)

mlflow.set_experiment(experiment_name)

# Load data
df = pd.read_csv('data/data.csv')

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df.species
print(f"Dataset loaded. {X.shape[0]} rows found.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Hyperparameter Search Space for Hyperopt
search_space = {
    'max_depth': hp.choice('max_depth', [None, 3, 4, 5]),
    'min_samples_split': hp.choice('min_samples_split', [3, 9, 11])
}

# Define the objective function for Hyperopt
def objective(params):
    with mlflow.start_run(nested=True):
        max_depth = params['max_depth']
        min_samples_split = int(params['min_samples_split'])
        
        # Train Model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        
        # Log Parameters and Metrics
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        
        return {'loss': -accuracy, 'status': STATUS_OK}

print("Starting Hyperopt optimization...")

experiment = mlflow.get_experiment_by_name("Iris_DecisionTree_Hyperopt")

with mlflow.start_run(run_name="Hyperopt_Optimization"):
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials
    )

print("Optimization complete!")
print("Searching for the best model to register...")
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_accuracy = best_run.data.metrics['accuracy']

print(f"Best run found! Run ID: {best_run_id} with Accuracy: {best_accuracy}")

model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri=model_uri, name="Iris-DecisionTree")

print("Best model successfully registered!")
