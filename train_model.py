"""Train and save a house-price regression pipeline."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
CATEGORICAL_FEATURES = ["location"]
TARGET = "price_inr"


def build_pipeline():
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("location", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])


def train_and_save_model(data_path="data/house_prices.csv", model_path="models/house_price_model.joblib"):
    data = pd.read_csv(data_path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].drop_duplicates().dropna().copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data = data[(data["area_sqft"] > 0) & (data[TARGET] > 0)]
    if len(data) < 10:
        raise ValueError("Please provide at least 10 valid rows for a train/test split.")

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae_inr": float(mean_absolute_error(y_test, predictions)),
        "rmse_inr": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the house-price prediction model.")
    parser.add_argument("--data", default="data/house_prices.csv", help="Path to the CSV dataset")
    parser.add_argument("--model", default="models/house_price_model.joblib", help="Output model path")
    args = parser.parse_args()
    _, results = train_and_save_model(args.data, args.model)
    print("Model training complete.")
    for metric, value in results.items():
        print(f"{metric}: {value:,.4f}" if isinstance(value, float) else f"{metric}: {value}")
