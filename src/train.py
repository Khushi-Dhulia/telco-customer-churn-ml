import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
)
from evaluate import evaluate_model
from preprocessing import build_preprocessor


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download the Kaggle Telco Customer Churn CSV and place it in data/."
        )

    df = pd.read_csv(DATA_PATH)

    # TotalCharges contains blank strings in the source dataset.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove rows without a target, if any.
    df = df.dropna(subset=["Churn"])

    return df


def build_models(preprocessor):
    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return logistic, random_forest


def main():
    print("Loading dataset...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["Churn"].value_counts())
    print("\nTarget distribution (%):")
    print((df["Churn"].value_counts(normalize=True) * 100).round(2))

    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected values found in Churn column.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fresh preprocessing objects avoid accidental reuse between estimators.
    logistic_preprocessor = build_preprocessor()
    rf_preprocessor = build_preprocessor()
    logistic, random_forest = build_models(
        logistic_preprocessor
    )
    # Replace the RF pipeline's preprocessor with its own instance.
    random_forest.named_steps["preprocessor"] = rf_preprocessor

    print("\nTraining Logistic Regression baseline...")
    logistic.fit(X_train, y_train)
    logistic_metrics = evaluate_model(logistic, X_test, y_test)

    print("Logistic Regression:")
    for key, value in logistic_metrics.items():
        print(f"  {key}: {value:.4f}")

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    print("\nRunning Random Forest 5-fold GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    rf_metrics = evaluate_model(best_model, X_test, y_test)

    print("\nBest parameters:")
    print(grid_search.best_params_)
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    print("\nRandom Forest test metrics:")
    for key, value in rf_metrics.items():
        print(f"  {key}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "churn_rate": float(y.mean()),
        },
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
        "random_forest_best_cv_roc_auc": float(grid_search.best_score_),
        "random_forest_best_params": grid_search.best_params_,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
