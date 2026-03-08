import pandas as pd
import pytest

DATA_PATH = "data/data.csv"
EVAL_PATH = "data/eval.csv"

@pytest.fixture
def data():
    return pd.read_csv(DATA_PATH)

@pytest.fixture
def eval_data():
    return pd.read_csv(EVAL_PATH)

def test_no_missing_values(data, eval_data):
    """Check for missing values in both datasets."""
    assert data.isnull().sum().sum() == 0, "Missing values found in training data!"
    assert eval_data.isnull().sum().sum() == 0, "Missing values found in eval data!"

def test_data_shape(data, eval_data):
    """Ensure both datasets have the expected minimum number of features."""
    for df in [data, eval_data]:
        assert df.shape[1] >= 5, f"Expected at least 5 columns, but got {df.shape[1]}"
        assert df.shape[0] > 0, "Dataset is empty!"

def test_reasonable_value_ranges(data, eval_data):
    """Check if numerical columns are within reasonable IRIS measurements (0 to 15 cm)."""
    for df in [data, eval_data]:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            assert df[col].between(0, 15).all(), f"Values in {col} are out of bounds"
