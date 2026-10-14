"""
Multiple Linear Regression on the 50 Startups dataset (scikit-learn + CRISP-DM).

Usage: python startups_multiple_linear_regression.py [path/to/50_Startups.csv]
If the CSV is missing, synthetic data with the same schema is generated.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
TARGET = "Profit"
NUM_COLS = ["R&D Spend", "Administration", "Marketing Spend"]
CAT_COLS = ["State"]


# ---------------------------------------------------------------------------
# Helper: OLS coefficient table (std errors, t-stats, p-values) with numpy/scipy
# ---------------------------------------------------------------------------
def ols_summary(X, y, names):
    """X: 2D array WITHOUT intercept column. Returns a DataFrame of OLS stats."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    n, k = Xb.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.inv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    return pd.DataFrame(
        {"coef": beta, "std_err": se, "t": t, "p_value": p},
        index=["Intercept"] + list(names),
    )


def vif(X_df):
    """Variance Inflation Factor for each column (multicollinearity check)."""
    out = {}
    for col in X_df.columns:
        others = X_df.drop(columns=col)
        r2 = LinearRegression().fit(others, X_df[col]).score(others, X_df[col])
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


# ---------------------------------------------------------------------------
# 1. BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------------
def business_understanding():
    print("=" * 70)
    print("1. BUSINESS UNDERSTANDING")
    print("=" * 70)
    print("Goal    : Explain and predict startup Profit from R&D, Administration,")
    print("          Marketing spend and State.")
    print("Task    : Multiple linear regression:")
    print("          Profit = b0 + b1*R&D + b2*Admin + b3*Marketing + b4*State + e")
    print("Success : Test R^2 >= 0.90, significant predictors, assumptions satisfied.\n")


# ---------------------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
def load_data(path):
    if path and os.path.exists(path):
        print(f"Loaded {path}")
        return pd.read_csv(path)
    print("CSV not found -> generating synthetic data with the same schema.")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 50
    df = pd.DataFrame({
        "R&D Spend": rng.uniform(0, 165_000, n),
        "Administration": rng.uniform(51_000, 182_000, n),
        "Marketing Spend": rng.uniform(0, 470_000, n),
        "State": rng.choice(["New York", "California", "Florida"], n),
    })
    df[TARGET] = (50_000 + 0.85 * df["R&D Spend"] + 0.03 * df["Marketing Spend"]
                  + rng.normal(0, 8_000, n))
    return df


def data_understanding(df):
    print("=" * 70)
    print("2. DATA UNDERSTANDING")
    print("=" * 70)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary:\n", df.describe().round(2))
    print("\nCorrelation matrix:\n", df[NUM_COLS + [TARGET]].corr().round(3))
    print("\nVIF (multicollinearity; > 5-10 is a concern):")
    print(vif(df[NUM_COLS]).round(2))
    print()


# ---------------------------------------------------------------------------
# 3. DATA PREPARATION
# ---------------------------------------------------------------------------
def data_preparation(df):
    print("=" * 70)
    print("3. DATA PREPARATION")
    print("=" * 70)
    df = df.drop_duplicates()
    X, y = df[NUM_COLS + CAT_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Scaling isn't needed for OLS (coefficients stay in original $ units),
    # so we only one-hot encode State. drop='first' avoids the dummy-variable trap.
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first"), CAT_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("State one-hot encoded (first category dropped).\n")
    return X_train, X_test, y_train, y_test, prep


# ---------------------------------------------------------------------------
# 4. MODELING
# ---------------------------------------------------------------------------
def modeling(prep, X_train, y_train):
    print("=" * 70)
    print("4. MODELING")
    print("=" * 70)

    # Model A: all features
    full = Pipeline([("prep", prep), ("lr", LinearRegression())]).fit(X_train, y_train)
    names = list(full.named_steps["prep"].get_feature_names_out())
    Xt = full.named_steps["prep"].transform(X_train)
    table = ols_summary(Xt, y_train.values, names)
    print("Model A: all features")
    print(table.round(4))

    # Model B: backward elimination (drop highest p-value > 0.05 until done)
    keep = names.copy()
    while True:
        idx = [names.index(c) for c in keep]
        tbl = ols_summary(Xt[:, idx], y_train.values, keep)
        pvals = tbl.drop(index="Intercept")["p_value"]
        if pvals.max() <= 0.05 or len(keep) == 1:
            break
        drop = pvals.idxmax()
        print(f"  Backward elimination: dropping '{drop}' (p={pvals.max():.3f})")
        keep.remove(drop)
    print("\nModel B: after backward elimination")
    print(tbl.round(4))
    print("Kept features:", keep, "\n")

    class _Subset:
        """Small wrapper so Model B behaves like a pipeline."""
        def __init__(self, prep, cols, lr):
            self.prep, self.cols, self.lr = prep, cols, lr
        def predict(self, X):
            Z = self.prep.transform(X)
            return self.lr.predict(Z[:, [names.index(c) for c in self.cols]])

    lr_b = LinearRegression().fit(Xt[:, [names.index(c) for c in keep]], y_train)
    reduced = _Subset(full.named_steps["prep"], keep, lr_b)
    return full, reduced, names


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
def evaluation(full, reduced, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("5. EVALUATION")
    print("=" * 70)

    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(full, X_train, y_train, cv=cv, scoring="r2")
    print(f"5-fold CV R^2 (full model): {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}\n")

    rows = []
    for name, model in [("Full model", full), ("Reduced model", reduced)]:
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Residual diagnostics on the test set (reduced model)
    pred = reduced.predict(X_test)
    resid = y_test.values - pred
    print("\nAssumption checks (test residuals):")
    print(f"  Mean residual (~0 expected)      : {resid.mean():,.1f}")
    sw_stat, sw_p = stats.shapiro(resid)
    print(f"  Shapiro-Wilk normality p-value   : {sw_p:.3f}  (> 0.05 = looks normal)")
    corr_abs = np.corrcoef(pred, np.abs(resid))[0, 1]
    print(f"  Corr(pred, |resid|) [heterosc.]  : {corr_abs:.3f}  (near 0 = constant variance)")
    print()


# ---------------------------------------------------------------------------
# 6. DEPLOYMENT
# ---------------------------------------------------------------------------
def deployment(full):
    import joblib
    print("=" * 70)
    print("6. DEPLOYMENT")
    print("=" * 70)
    joblib.dump(full, "startup_mlr_model.joblib")
    print("Saved full-model pipeline -> startup_mlr_model.joblib")
    new = pd.DataFrame([{
        "R&D Spend": 120_000, "Administration": 100_000,
        "Marketing Spend": 250_000, "State": "California",
    }])
    loaded = joblib.load("startup_mlr_model.joblib")
    print(f"Example prediction: ${loaded.predict(new)[0]:,.0f}")
    print("Next steps: serve via an API, monitor residuals/drift, retrain with new data.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "50_Startups.csv"
    business_understanding()
    df = load_data(path)
    data_understanding(df)
    X_train, X_test, y_train, y_test, prep = data_preparation(df)
    full, reduced, names = modeling(prep, X_train, y_train)
    evaluation(full, reduced, X_train, y_train, X_test, y_test)
    deployment(full)
