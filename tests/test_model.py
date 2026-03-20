import pandas as pd
import pytest
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import os

EVAL_PATH = "data/eval.csv"
MODEL_URI = "models:/Iris-DecisionTree/latest"

@pytest.fixture
def eval_data():
    return pd.read_csv(EVAL_PATH)

@pytest.fixture
def model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    return mlflow.sklearn.load_model(MODEL_URI)

def test_model_performance(eval_data, model):
    target_col = eval_data.columns[-1]
    
    X_test = eval_data.drop(columns=[target_col])
    y_test = eval_data[target_col]
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    with open("metrics.txt", "w") as f:
        f.write(f"**MLflow Decision Tree Accuracy:** {accuracy}\n")
    
    assert accuracy > 0.85, f"Model accuracy {accuracy} is below the 0.85 threshold!"
