"""
Trains a Linear Regression model to predict house prices from the
Hyderabad house price dataset (data/house_prices.csv).

Pipeline:
    - One-hot encode the 'area' column
    - Scale numeric features
    - Fit Linear Regression
    - Evaluate on a held-out test set
    - Save the fitted pipeline to model/house_price_model.joblib

Run:
    python train_model.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/house_prices.csv"
MODEL_PATH = "model/house_price_model.joblib"

NUMERIC_FEATURES = ["sqft", "bhk", "bathrooms", "age_years"]
CATEGORICAL_FEATURES = ["area"]
TARGET = "price_lakhs"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model evaluation on held-out test set:")
    print(f"  MAE  : {mae:.2f} Lakhs")
    print(f"  RMSE : {rmse:.2f} Lakhs")
    print(f"  R^2  : {r2:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
