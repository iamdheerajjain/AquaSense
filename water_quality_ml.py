import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RANDOM_STATE = 42
MODEL_PATH_DEFAULT = "model.joblib"

def split_features_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features


def evaluate_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    roc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None and len(np.unique(y_true)) == 2 else None
    print("\n=== Classification Metrics ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC:       {roc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print("\n=== Regression Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")


def cross_validate(model, X, y, task, n_splits=5):
    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1_weighted"
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_root_mean_squared_error"

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n{n_splits}-Fold CV ({scoring}): {scores.mean():.4f} ± {scores.std():.4f}")


def plot_feature_importance(model, feature_names, top_k=20, title="Feature Importance"):

    if not hasattr(model, "feature_importances_"):
        print("Model has no `feature_importances_`. Skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


def get_model(task: str, model_type: str):
    if task == "classification":
        if model_type == "logreg":
            return LogisticRegression(max_iter=1000, n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None)
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            raise ValueError("Unknown model_type for classification. Choose from: logreg, rf")

    elif task == "regression":
        if model_type == "ridge":
            return Ridge(random_state=RANDOM_STATE)
        elif model_type == "rf":
            return RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            raise ValueError("Unknown model_type for regression. Choose from: ridge, rf")

    else:
        raise ValueError("task must be 'classification' or 'regression'")


def main():
    parser = argparse.ArgumentParser(description="Water Quality Prediction")
    parser.add_argument("--csv", type=str, default=None, help="Training CSV path")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")
    parser.add_argument("--model_type", type=str, default=None,
                        help="classification: [logreg, rf], regression: [ridge, rf]. Defaults: rf for both.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH_DEFAULT, help="Where to save the trained model")
    parser.add_argument("--predict_csv", type=str, default=None, help="CSV with new samples to predict")
    args = parser.parse_args()

    if args.predict_csv is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")
        print(f"Loading model from {args.model_path}")
        pipe = joblib.load(args.model_path)
        new_df = pd.read_csv(args.predict_csv)
        preds = pipe.predict(new_df)
        # If classifier and has predict_proba
        proba = pipe.predict_proba(new_df)[:, 1] if hasattr(pipe, "predict_proba") else None

        print("\n=== Predictions ===")
        out = new_df.copy()
        out["prediction"] = preds
        if proba is not None:
            out["probability_of_1"] = proba
        print(out.head(20))
        out.to_csv("predictions.csv", index=False)
        print("\nSaved predictions to predictions.csv")
        return

    if args.csv is None or args.target is None:
        parser.error("--csv and --target are required for training.")

    df = pd.read_csv(args.csv)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    X, y = split_features_target(df, args.target)
    preprocessor, num_feats, cat_feats = build_preprocessor(X)
    print(f"Numeric features: {len(num_feats)}, Categorical features: {len(cat_feats)}")

    if args.model_type is None:
        args.model_type = "rf" if args.task in ["classification", "regression"] else None

    base_model = get_model(args.task, args.model_type)

    # Build full pipeline
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", base_model)
    ])

    stratify = y if args.task == "classification" and len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    cross_validate(pipe, X_train, y_train, task=args.task, n_splits=5)

    print("\nTraining...")
    pipe.fit(X_train, y_train)

    print("\nEvaluating on hold-out test set...")
    y_pred = pipe.predict(X_test)

    if args.task == "classification":
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        evaluate_classification(y_test, y_pred, y_proba)
    else:
        evaluate_regression(y_test, y_pred)

    joblib.dump(pipe, args.model_path)
    print(f"\nModel saved to {args.model_path}")

    model = pipe.named_steps["model"]
    try:
        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        cat_new_names = ohe.get_feature_names_out(cat_feats)
        final_feature_names = np.r_[num_feats, cat_new_names]
    except Exception:
        final_feature_names = [f"f_{i}" for i in range(len(model.feature_importances_))] \
            if hasattr(model, "feature_importances_") else None

    if hasattr(model, "feature_importances_") and final_feature_names is not None:
        plot_feature_importance(model, final_feature_names, top_k=20,
                                title=f"Top 20 Feature Importances ({args.model_type})")


if __name__ == "__main__":
    main()
