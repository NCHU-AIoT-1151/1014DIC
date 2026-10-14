# ============================================================
# BOSTON HOUSING - CRISP-DM MULTIPLE LINEAR REGRESSION
#
# Model B: 13 Features + Intercept
#
# Feature Selection:
# 1. LassoCV
# 2. RFECV
# 3. Sequential Forward Selection (SFS)
# 4. ElasticNetCV
# 5. Exhaustive Feature Selection (limited to k <= 6 for speed)
#
# pip install numpy pandas matplotlib scikit-learn statsmodels scipy joblib
# Usage: python main.py [boston.csv]
# ============================================================

import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "MEDV"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


# ------------------------------------------------------------
# 1. BUSINESS UNDERSTANDING
# ------------------------------------------------------------
section("1. BUSINESS UNDERSTANDING")
print("Goal: predict MEDV (median house value $1000s) from 13 Boston district features")
print("Method: Multiple Linear Regression (CRISP-DM workflow)")
print("Note: scikit-learn removed load_boston; uses external CSV or synthetic fallback")


# ------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ------------------------------------------------------------
section("2. DATA UNDERSTANDING")


def load_data(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if df.columns[0].lower() in ("", "unnamed: 0"):
            df = df.iloc[:, 1:]
        df.columns = [c.upper() for c in df.columns]
        if TARGET not in df.columns:
            raise ValueError(f"Column {TARGET!r} not found. Available: {list(df.columns)}")
        print(f"Loaded real data: {path}")
        return df
    print("CSV not found -> using synthetic demo data (same schema as Boston Housing)")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 506
    crim  = rng.exponential(3.6, n)
    zn    = rng.choice([0.0, 20.0, 80.0], n, p=[0.7, 0.2, 0.1])
    indus = rng.uniform(0.5, 27.7, n)
    chas  = rng.choice([0.0, 1.0], n, p=[0.93, 0.07])
    nox   = rng.uniform(0.38, 0.87, n)
    rm    = rng.normal(6.28, 0.70, n)
    age   = rng.uniform(2.9, 100.0, n)
    dis   = rng.exponential(3.8, n).clip(1.0, 12.0)
    rad   = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 24.0], n)
    tax   = rad * 30 + rng.normal(280, 40, n)
    ptrat = rng.uniform(12.6, 22.0, n)
    b     = rng.uniform(0.3, 396.9, n)
    lstat = rng.uniform(1.7, 37.9, n)
    medv  = (50 - 0.12*crim + 0.04*zn - 0.02*indus + 2.5*chas
             - 18*nox + 4.5*rm - 0.01*age - 1.2*dis
             + 0.3*rad - 0.01*tax - 1.0*ptrat + 0.01*b
             - 0.5*lstat + rng.normal(0, 3.5, n))
    medv = np.clip(medv, 5.0, 50.0)
    return pd.DataFrame({
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas,
        "NOX": nox, "RM": rm, "AGE": age, "DIS": dis,
        "RAD": rad, "TAX": tax, "PTRATIO": ptrat, "B": b,
        "LSTAT": lstat, TARGET: medv,
    })


csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(OUT_DIR), "data", "boston.csv")

raw = load_data(csv_path)
print("\nShape:", raw.shape)
print("\nMissing values:\n", raw.isna().sum())
print("\nDescriptive statistics:")
print(raw.describe().round(2))
corr = raw.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(f"\nCorrelation with {TARGET}:\n{corr.round(3)}")


# ------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------
section("3. DATA PREPARATION")
df = raw.dropna().copy()
print(f"Rows after dropna: {len(df)} (dropped {len(raw) - len(df)})")
X = df.drop(columns=TARGET)
y = df[TARGET]
features = list(X.columns)
print("Features (13):", features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
Xtr = pd.DataFrame(scaler.transform(X_train), columns=features, index=X_train.index)
Xte = pd.DataFrame(scaler.transform(X_test),  columns=features, index=X_test.index)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(cols, label):
    m = LinearRegression()
    cv_r2   = cross_val_score(m, Xtr[cols], y_train, cv=cv, scoring="r2").mean()
    cv_rmse = np.sqrt(-cross_val_score(
        m, Xtr[cols], y_train, cv=cv, scoring="neg_mean_squared_error").mean())
    m.fit(Xtr[cols], y_train)
    pred = m.predict(Xte[cols])
    return {
        "Method": label, "n_features": len(cols), "Features": ", ".join(cols),
        "CV_R2": cv_r2, "CV_RMSE": cv_rmse,
        "Test_R2": r2_score(y_test, pred),
        "Test_MAE": mean_absolute_error(y_test, pred),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }, pred


# ------------------------------------------------------------
# 4. BASELINE + VIF
# ------------------------------------------------------------
section("4. BASELINE & VIF")
base = DummyRegressor(strategy="mean").fit(Xtr, y_train)
base_pred = base.predict(Xte)
baseline_row = {
    "Method": "Baseline (mean)", "n_features": 0, "Features": "-",
    "CV_R2": cross_val_score(DummyRegressor(), Xtr, y_train, cv=cv, scoring="r2").mean(),
    "CV_RMSE": float(np.std(y_train.values)),
    "Test_R2": r2_score(y_test, base_pred),
    "Test_MAE": mean_absolute_error(y_test, base_pred),
    "Test_RMSE": np.sqrt(mean_squared_error(y_test, base_pred)),
}
print(f"Baseline Test R2 = {baseline_row['Test_R2']:.3f}, RMSE = {baseline_row['Test_RMSE']:.3f}")

Xc = sm.add_constant(X_train)
vif = pd.Series(
    [variance_inflation_factor(Xc.values.astype(float), i) for i in range(1, Xc.shape[1])],
    index=features)
print("\nVIF (>5~10 indicates multicollinearity):")
print(vif.round(2).to_string())
high_vif = vif[vif > 5]
if len(high_vif):
    print(f"\n[WARNING] High VIF features: {list(high_vif.index)}")


# ------------------------------------------------------------
# 5. MODELING
# ------------------------------------------------------------
section("5. MODEL B: 13 FEATURES + INTERCEPT (statsmodels OLS)")
ols = sm.OLS(y_train, sm.add_constant(X_train.astype(float))).fit()
print(ols.summary().tables[1])
print(f"R2={ols.rsquared:.4f} | Adj R2={ols.rsquared_adj:.4f} | AIC={ols.aic:.1f} | BIC={ols.bic:.1f}")

section("5. FEATURE SELECTION (5 methods)")
selected = {}

lasso = LassoCV(cv=cv, max_iter=200000, random_state=RANDOM_STATE).fit(Xtr, y_train)
selected["1. LassoCV"] = [f for f, c in zip(features, lasso.coef_) if abs(c) > 1e-8]
print(f"LassoCV alpha={lasso.alpha_:.4f} -> {selected['1. LassoCV']}")

rfecv = RFECV(LinearRegression(), cv=cv, scoring="r2").fit(Xtr, y_train)
selected["2. RFECV"] = [f for f, s in zip(features, rfecv.support_) if s]
print(f"RFECV n_features={rfecv.n_features_} -> {selected['2. RFECV']}")

sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select="auto", tol=0.002,
    direction="forward", cv=cv, scoring="r2").fit(Xtr, y_train)
selected["3. SFS (forward)"] = [f for f, s in zip(features, sfs.get_support()) if s]
print(f"SFS -> {selected['3. SFS (forward)']}")

enet = ElasticNetCV(cv=cv, l1_ratio=[.1, .3, .5, .7, .9, 1],
                    max_iter=200000, random_state=RANDOM_STATE).fit(Xtr, y_train)
selected["4. ElasticNetCV"] = [f for f, c in zip(features, enet.coef_) if abs(c) > 1e-8]
print(f"ElasticNetCV alpha={enet.alpha_:.4f}, l1={enet.l1_ratio_} -> {selected['4. ElasticNetCV']}")

MAX_K = 6
print(f"\nExhaustive Search (k<={MAX_K}) ...")
best_cols, best_cv_score = None, -np.inf
for k in range(1, MAX_K + 1):
    for combo in itertools.combinations(features, k):
        s = cross_val_score(LinearRegression(), Xtr[list(combo)], y_train,
                            cv=cv, scoring="r2").mean()
        if s > best_cv_score + 1e-9:
            best_cv_score, best_cols = s, list(combo)
selected["5. Exhaustive"] = best_cols
print(f"Exhaustive best (CV R2={best_cv_score:.4f}) -> {best_cols}")

for name, cols in selected.items():
    print(f"  {name:<22} -> {cols}")


# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------
section("6. EVALUATION")
rows, preds = [baseline_row], {"Baseline (mean)": base_pred}
row, p = evaluate(features, "Model B (all 13)")
rows.append(row); preds["Model B (all 13)"] = p
for name, cols in selected.items():
    row, p = evaluate(cols, name)
    rows.append(row); preds[name] = p

report = pd.DataFrame(rows)
pd.set_option("display.width", 230, "display.max_colwidth", 60)
print(report.round(4).to_string(index=False))
report.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

best_name = report.iloc[1:].sort_values("Test_R2", ascending=False).iloc[0]["Method"]
print(f"\nBest Test R2: {best_name}")

resid = y_test.values - preds["Model B (all 13)"]
print("\nResidual diagnostics (Model B, test set):")
print(f"  Mean residual : {resid.mean():.4f}")
print(f"  Residual std  : {resid.std():.4f}")
sw_stat, sw_p = stats.shapiro(resid)
print(f"  Shapiro-Wilk p = {sw_p:.4f}  ({'approx. normal' if sw_p >= 0.05 else 'non-normal'})")


# ------------------------------------------------------------
# 7. PLOTS
# ------------------------------------------------------------
section("7. PLOTS")
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
pred_b = preds["Model B (all 13)"]
lo, hi = float(y_test.min()) * 0.9, float(y_test.max()) * 1.05

ax[0, 0].scatter(y_test, pred_b, edgecolor="k", alpha=0.8)
ax[0, 0].plot([lo, hi], [lo, hi], "r--", label="Perfect fit")
ax[0, 0].set(title=f"Model B: Actual vs Predicted (R2={r2_score(y_test, pred_b):.3f})",
             xlabel="Actual MEDV ($1000s)", ylabel="Predicted MEDV ($1000s)",
             xlim=(lo, hi), ylim=(lo, hi))
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

ax[0, 1].scatter(pred_b, resid, edgecolor="k", alpha=0.8)
ax[0, 1].axhline(0, color="r", ls="--")
ax[0, 1].set(title="Residuals vs Predicted",
             xlabel="Predicted MEDV", ylabel="Residual")
ax[0, 1].grid(alpha=.3)

stats.probplot(resid, dist="norm", plot=ax[1, 0])
ax[1, 0].set_title("Residual Q-Q Plot"); ax[1, 0].grid(alpha=.3)

plot_df = report.iloc[1:].set_index("Method")[["CV_R2", "Test_R2"]]
plot_df.plot.barh(ax=ax[1, 1])
ax[1, 1].set(title="CV R2 vs Test R2 by Method", xlabel="R2")
ax[1, 1].set_xlim(max(0, float(plot_df.min().min()) - 0.05), 1.0)
ax[1, 1].grid(alpha=.3)
fig.tight_layout()
savefig(fig, "main_plots.png")
plt.close(fig)

top_features = corr.abs().sort_values(ascending=False).head(6).index.tolist()
fig2, axs = plt.subplots(2, 3, figsize=(18, 9))
for a, c in zip(axs.flat, top_features):
    a.scatter(df[c], df[TARGET], edgecolor="k", alpha=0.6, s=15)
    m_, b_ = np.polyfit(df[c], df[TARGET], 1)
    xs = np.linspace(float(df[c].min()), float(df[c].max()), 100)
    a.plot(xs, m_*xs + b_, "r-", lw=1.5)
    r = np.corrcoef(df[c], df[TARGET])[0, 1]
    a.set(xlabel=c, ylabel="MEDV", title=f"{c} vs MEDV (r={r:.2f})")
    a.grid(alpha=.3)
fig2.tight_layout()
savefig(fig2, "main_feature_scatter.png")
plt.close(fig2)

lasso_order = [f for f, c in sorted(zip(features, lasso.coef_),
                                    key=lambda t: abs(t[1]), reverse=True)]
ks = list(range(1, len(features) + 1))
rmse_k, r2_k = [], []
for k in ks:
    cols_k = lasso_order[:k]
    m_k = LinearRegression().fit(Xtr[cols_k], y_train)
    p_k = m_k.predict(Xte[cols_k])
    rmse_k.append(float(np.sqrt(mean_squared_error(y_test, p_k))))
    r2_k.append(float(r2_score(y_test, p_k)))

fig3, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(ks, rmse_k, "o-", color="#4f46e5")
a1.set(title="Test RMSE vs k (Lasso order)", xlabel="k features", ylabel="RMSE")
a1.axvline(int(np.argmin(rmse_k)) + 1, color="r", ls="--", label="Min RMSE")
a1.legend(); a1.grid(alpha=.3)
a2.plot(ks, r2_k, "o-", color="#16a34a")
a2.set(title="Test R2 vs k (Lasso order)", xlabel="k features", ylabel="R2")
a2.axvline(int(np.argmax(r2_k)) + 1, color="r", ls="--", label="Max R2")
a2.legend(); a2.grid(alpha=.3)
fig3.tight_layout()
savefig(fig3, "rmse_r2_vs_k.png")
plt.close(fig3)


# ------------------------------------------------------------
# 8. TOP 10 OUTLIERS
# ------------------------------------------------------------
section("8. TOP 10 OUTLIERS (Model B, full data)")
full_ols = sm.OLS(y.astype(float), sm.add_constant(X.astype(float))).fit()
infl = full_ols.get_influence()
out_df = pd.DataFrame({
    "Actual": y, "Predicted": full_ols.fittedvalues,
    "Residual": full_ols.resid,
    "Studentized_Resid": infl.resid_studentized_external,
    "Cooks_D": infl.cooks_distance[0],
    "Leverage": infl.hat_matrix_diag,
})
top10 = out_df.reindex(
    out_df["Studentized_Resid"].abs().sort_values(ascending=False).index).head(10)
print(top10.round(3).to_string())
top10.to_csv(os.path.join(OUT_DIR, "top10_outliers.csv"))
threshold = 4 / len(y)
print(f"\n|Studentized Resid|>3 -> outlier; Cooks D > {threshold:.4f} (4/n) -> influential")
print(f"High-influence points: {(infl.cooks_distance[0] > threshold).sum()}")


# ------------------------------------------------------------
# 9. DEPLOYMENT
# ------------------------------------------------------------
section("9. DEPLOYMENT")
import joblib
final_cols = selected.get(best_name, features)
final_model = LinearRegression().fit(X[final_cols].astype(float), y)
joblib.dump(
    {"model": final_model, "features": final_cols, "scaler": scaler},
    os.path.join(OUT_DIR, "boston_mlr_final.joblib"))
print(f"Model saved ({best_name}), features: {final_cols}")

Xall_scaled = pd.DataFrame(scaler.transform(X[final_cols]), columns=final_cols)
coef_model = LinearRegression().fit(Xall_scaled, y)
coef_df = pd.Series(coef_model.coef_, index=final_cols).sort_values(key=abs, ascending=False)
print("\nStandardized coefficients (final model):")
print(coef_df.round(4).to_string())
