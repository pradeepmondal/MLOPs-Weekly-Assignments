import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv('data/data.csv')

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df.species
print(f"Dataset loaded. {X.shape[0]} rows found.")

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X,y)
print("Model trained successfully.")

joblib.dump(mod_dt, "model.joblib")
print("Model saved locally.")