import pandas as pd
import pytest
import joblib 
from sklearn.metrics import accuracy_score

EVAL_PATH = "data/eval.csv"
MODEL_PATH = "model.joblib"

@pytest.fixture
def eval_data():
    return pd.read_csv(EVAL_PATH)

@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)

def test_model_performance(eval_data, model):
    """Evaluate model directly on the eval dataset and assert metrics."""
    target_col = eval_data.columns[-1]
    
    X_test = eval_data.drop(columns=[target_col])
    y_test = eval_data[target_col]
    
    predictions = model.predict(X_test)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Write metrics to a file so CML can report it in the PR
    with open("metrics.txt", "w") as f:
        f.write(f"**Model Accuracy on Eval Set:** {accuracy}\n")
    
    # Assert threshold (Fail CI if accuracy drops below 85%)
    assert accuracy > 0.85, f"Model accuracy {accuracy} is below the 0.85 threshold!"
