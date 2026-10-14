"""
50 Startups profit prediction with scikit-learn, structured by CRISP-DM.

Dataset columns: R&D Spend, Administration, Marketing Spend, State, Profit
Usage: python startups_crisp_dm.py [path/to/50_Startups.csv]
If no CSV is found, a synthetic dataset with the same schema is generated
so the script still runs end to end.
"""
import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
TARGET = "Profit"
NUM_COLS = ["R&D Spend", "Administration", "Marketing Spend"]
CAT_COLS = ["State"]


# ---------------------------------------------------------------------------
# 1. BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------------
def business_understanding():
    print("=" * 70)
    print("1. BUSINESS UNDERSTANDING")
    print("=" * 70)
    print("Goal    : Predict a startup's annual Profit from its spending and location,")
    print("          so investors can see which spend categories drive profit.")
    print("Task    : Supervised regression (target = Profit).")
    print("Success : R^2 >= 0.90 on held-out data and a low RMSE (in $).")
    print("          Models must also be interpretable enough to advise on spend.\n")


# ---------------------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
def load_data(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {path}")
    else:
        print("CSV not found -> generating synthetic data with the same schema.")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 50
        df = pd.DataFrame({
            "R&D Spend": rng.uniform(0, 165_000, n),
            "Administration": rng.uniform(51_000, 182_000, n),
            "Marketing Spend": rng.uniform(0, 470_000, n),
            "State": rng.choice(["New York", "California", "Florida"], n),
        })
        df[TARGET] = (
            50_000
            + 0.85 * df["R&D Spend"]
            + 0.03 * df["Marketing Spend"]
            + rng.normal(0, 8_000, n)
        )
    return df


def data_understanding(df):
    print("=" * 70)
    print("2. DATA UNDERSTANDING")
    print("=" * 70)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary statistics:\n", df.describe().round(2))
    print("\nState counts:\n", df["State"].value_counts())
    print("\nCorrelation with Profit (numeric features):")
    print(df[NUM_COLS + [TARGET]].corr()[TARGET].sort_values(ascending=False).round(3))
    print("\nObservation: R&D Spend is usually the strongest predictor;")
    print("Administration is usually weak. Zero-spend rows may exist (valid values).\n")


# ---------------------------------------------------------------------------
# 3. DATA PREPARATION
# ---------------------------------------------------------------------------
def data_preparation(df):
    print("=" * 70)
    print("3. DATA PREPARATION")
    print("=" * 70)
    df = df.drop_duplicates().copy()
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Preprocessing lives inside the pipeline -> no data leakage during CV.
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        # drop='first' avoids the dummy-variable trap for linear models
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ])
    print("Preprocessing: StandardScaler (numeric) + OneHotEncoder (State)\n")
    return X_train, X_test, y_train, y_test, preprocessor


# ---------------------------------------------------------------------------
# 4. MODELING
# ---------------------------------------------------------------------------
def modeling(preprocessor, X_train, y_train):
    print("=" * 70)
    print("4. MODELING")
    print("=" * 70)
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10_000),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        results[name] = (scores.mean(), scores.std())
        print(f"{name:<18} CV R^2 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nBest by CV: {best_name}")

    # Light hyperparameter tuning for the regularised linear models
    print("\nTuning Ridge alpha with GridSearchCV ...")
    ridge_pipe = Pipeline([("prep", preprocessor), ("model", Ridge())])
    grid = GridSearchCV(
        ridge_pipe,
        {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        cv=cv, scoring="r2",
    )
    grid.fit(X_train, y_train)
    print("Best Ridge params:", grid.best_params_, f"| CV R^2 = {grid.best_score_:.3f}")

    # Fit everything on the full training set for evaluation
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        fitted[name] = pipe.fit(X_train, y_train)
    fitted["Ridge (tuned)"] = grid.best_estimator_
    print()
    return fitted


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
def evaluation(fitted, X_test, y_test):
    print("=" * 70)
    print("5. EVALUATION (hold-out test set)")
    print("=" * 70)
    rows = []
    for name, pipe in fitted.items():
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        })
    report = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print(report.round(3).to_string(index=False))

    best = report.loc[0, "Model"]
    print(f"\nBest on test set: {best}")
    ok = report.loc[0, "R2"] >= 0.90
    print(f"Business criterion (R^2 >= 0.90): {'MET' if ok else 'NOT MET'}")

    # Interpretation from the linear model
    lin = fitted["LinearRegression"]
    feat_names = lin.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(lin.named_steps["model"].coef_, index=feat_names)
    print("\nLinear coefficients (standardised numeric features):")
    print(coefs.sort_values(key=abs, ascending=False).round(1))

    rf = fitted["RandomForest"]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=feat_names)
    print("\nRandomForest feature importances:")
    print(imp.sort_values(ascending=False).round(3))
    print()
    return best


# ---------------------------------------------------------------------------
# 6. DEPLOYMENT
# ---------------------------------------------------------------------------
def deployment(fitted, best_name):
    import joblib
    print("=" * 70)
    print("6. DEPLOYMENT")
    print("=" * 70)
    model = fitted[best_name]
    joblib.dump(model, "startup_profit_model.joblib")
    print(f"Saved '{best_name}' pipeline -> startup_profit_model.joblib")

    # Example of scoring a new startup with the saved pipeline
    loaded = joblib.load("startup_profit_model.joblib")
    new = pd.DataFrame([{
        "R&D Spend": 120_000, "Administration": 100_000,
        "Marketing Spend": 250_000, "State": "California",
    }])
    print(f"Example prediction: ${loaded.predict(new)[0]:,.0f}")
    print("Next steps: wrap in an API (FastAPI/Flask), monitor drift, retrain periodically.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "50_Startups.csv"

    business_understanding()
    df = load_data(csv_path)
    data_understanding(df)
    X_train, X_test, y_train, y_test, prep = data_preparation(df)
    fitted = modeling(prep, X_train, y_train)
    best = evaluation(fitted, X_test, y_test)
    deployment(fitted, best)
