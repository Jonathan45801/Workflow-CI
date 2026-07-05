import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
import sys

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_dataset_processing.csv")
    data = pd.read_csv(file_path)
    x = data.drop(columns="rating",axis=1)
    y = data["rating"]
    X_train, X_test, y_train, y_test,= train_test_split(
        x,
        y,
        random_state=42,
        test_size=0.2
    )

    object_cols = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols =x.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor= ColumnTransformer(
        transformers=[
            ("cat",OneHotEncoder(handle_unknown="ignore"),object_cols)
        ],
        remainder="passthrough"
    )
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    # Full pipeline: raw data in (strings included) -> encode -> model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=505, max_depth=37))
    ])

    input_example=X_train.iloc[0:5]

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        predicted_qualities = pipeline.predict(X_test)

        mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=input_example
        )
        
        # Log metrics
        accuracy = pipeline.score(X_test, y_test)
        mlflow.log_metric("r2_score", accuracy)