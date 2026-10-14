# ============================================================
# 50 STARTUPS - CRISP-DM MULTIPLE LINEAR REGRESSION
#
# Model B: 5 Features + Intercept
#
# Feature Selection:
# 1. LassoCV
# 2. RFECV
# 3. Sequential Forward Selection (SFS)
# 4. ElasticNetCV
# 5. Exhaustive Feature Selection
#
# Plus: Baseline, VIF, CV, Metrics, Plots, Top 10 Outliers
#
# pip install numpy pandas matplotlib scikit-learn statsmodels
# ============================================================

import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 本機想跳出視窗就刪掉這行
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor  # noqa: F401 (備用)
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "Profit"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# 1. BUSINESS UNDERSTANDING
# ------------------------------------------------------------
section("1. BUSINESS UNDERSTANDING")
print("目標: 用 R&D / Administration / Marketing / State 預測新創公司 Profit")
print("方法: 多元線性迴歸 (Multiple Linear Regression)")
print("成功標準: Test R² >= 0.90, 並找出最精簡且有解釋力的特徵組合")


# ------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ------------------------------------------------------------
section("2. DATA UNDERSTANDING")


def load_data(path):
    if path and os.path.exists(path):
        print(f"Loaded {path}")
        return pd.read_csv(path)
    print("找不到 CSV -> 使用模擬資料 (欄位相同)")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 50
    d = pd.DataFrame({
        "R&D Spend": rng.uniform(0, 165_000, n),
        "Administration": rng.uniform(51_000, 182_000, n),
        "Marketing Spend": rng.uniform(0, 470_000, n),
        "State": rng.choice(["New York", "California", "Florida"], n),
    })
    d[TARGET] = (50_000 + 0.85 * d["R&D Spend"] + 0.03 * d["Marketing Spend"]
                 + rng.normal(0, 8_000, n))
    return d


raw = load_data(sys.argv[1] if len(sys.argv) > 1 else "50_Startups.csv")
print("Shape:", raw.shape)
print("Missing values:\n", raw.isna().sum())
print(raw.describe().round(2))


# ------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------
section("3. DATA PREPARATION")
df = pd.get_dummies(raw, columns=["State"], drop_first=True, dtype=float)
X = df.drop(columns=TARGET)          # 5 features: 3 numeric + 2 State dummies
y = df[TARGET]
features = list(X.columns)
print("Features (Model B):", features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
Xtr = pd.DataFrame(scaler.transform(X_train), columns=features, index=X_train.index)
Xte = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(cols, label):
    """用指定特徵擬合 OLS,回傳 CV R² 與 Test 指標與預測值。"""
    m = LinearRegression()
    cv_r2 = cross_val_score(m, Xtr[cols], y_train, cv=cv, scoring="r2").mean()
    m.fit(Xtr[cols], y_train)
    pred = m.predict(Xte[cols])
    return {
        "Method": label,
        "n_features": len(cols),
        "Features": ", ".join(cols),
        "CV_R2": cv_r2,
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
    "Test_R2": r2_score(y_test, base_pred),
    "Test_MAE": mean_absolute_error(y_test, base_pred),
    "Test_RMSE": np.sqrt(mean_squared_error(y_test, base_pred)),
}
print(f"Baseline Test R² = {baseline_row['Test_R2']:.3f}, RMSE = {baseline_row['Test_RMSE']:,.0f}")

Xc = sm.add_constant(X_train)
vif = pd.Series(
    [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
    index=features)
print("\nVIF (> 5~10 表示共線性問題):")
print(vif.round(2))


# ------------------------------------------------------------
# 5. MODELING
# ------------------------------------------------------------
section("5. MODEL B: 5 FEATURES + INTERCEPT (statsmodels OLS)")
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(ols.summary().tables[1])
print(f"R² = {ols.rsquared:.4f} | Adj R² = {ols.rsquared_adj:.4f} | "
      f"AIC = {ols.aic:.1f} | BIC = {ols.bic:.1f}")

section("5. FEATURE SELECTION (5 methods)")
selected = {}

# (1) LassoCV: 係數不為 0 的特徵
lasso = LassoCV(cv=cv, max_iter=100_000, random_state=RANDOM_STATE).fit(Xtr, y_train)
selected["1. LassoCV"] = [f for f, c in zip(features, lasso.coef_) if abs(c) > 1e-8]
print(f"LassoCV alpha = {lasso.alpha_:.2f}")

# (2) RFECV
rfecv = RFECV(LinearRegression(), cv=cv, scoring="r2").fit(Xtr, y_train)
selected["2. RFECV"] = [f for f, s in zip(features, rfecv.support_) if s]

# (3) Sequential Forward Selection: 自動決定特徵數 (改善 < tol 就停)
sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select="auto", tol=0.002,
    direction="forward", cv=cv, scoring="r2").fit(Xtr, y_train)
selected["3. SFS (forward)"] = [f for f, s in zip(features, sfs.get_support()) if s]

# (4) ElasticNetCV
enet = ElasticNetCV(cv=cv, l1_ratio=[.1, .3, .5, .7, .9, 1], max_iter=100_000,
                    random_state=RANDOM_STATE).fit(Xtr, y_train)
selected["4. ElasticNetCV"] = [f for f, c in zip(features, enet.coef_) if abs(c) > 1e-8]
print(f"ElasticNetCV alpha = {enet.alpha_:.2f}, l1_ratio = {enet.l1_ratio_}")

# (5) Exhaustive: itertools 列出全部 2^5 - 1 = 31 種組合, 取 CV R² 最高者
best_cols, best_cv = None, -np.inf
for k in range(1, len(features) + 1):
    for combo in itertools.combinations(features, k):
        s = cross_val_score(LinearRegression(), Xtr[list(combo)], y_train,
                            cv=cv, scoring="r2").mean()
        if s > best_cv + 1e-9:       # 同分時偏好較少特徵
            best_cv, best_cols = s, list(combo)
selected["5. Exhaustive"] = best_cols

for name, cols in selected.items():
    print(f"{name:<18}-> {cols}")


# ------------------------------------------------------------
# 6. EVALUATION (CV + Metrics)
# ------------------------------------------------------------
section("6. EVALUATION")
rows, preds = [baseline_row], {"Baseline (mean)": base_pred}
row, p = evaluate(features, "Model B (all 5)")
rows.append(row); preds["Model B (all 5)"] = p
for name, cols in selected.items():
    row, p = evaluate(cols, name)
    rows.append(row); preds[name] = p

report = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_colwidth", 55)
print(report.round(3).to_string(index=False))
report.to_csv("model_comparison.csv", index=False)

best_name = report.iloc[1:].sort_values("Test_R2", ascending=False).iloc[0]["Method"]
print(f"\nTest R² 最高: {best_name}")

# 殘差診斷 (Model B)
resid = y_test.values - preds["Model B (all 5)"]
print("\n殘差診斷 (Model B, test set):")
print(f"  平均殘差: {resid.mean():,.1f}")
print(f"  Shapiro-Wilk p = {stats.shapiro(resid)[1]:.3f}  (> 0.05 ≈ 常態)")


# ------------------------------------------------------------
# 7. PLOTS
# ------------------------------------------------------------
section("7. PLOTS")
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
pred_b = preds["Model B (all 5)"]
lo, hi = y_test.min() * 0.95, y_test.max() * 1.05

# (a) Actual vs Predicted scatter
ax[0, 0].scatter(y_test, pred_b, edgecolor="k", alpha=0.85)
ax[0, 0].plot([lo, hi], [lo, hi], "r--", label="Perfect")
ax[0, 0].set(title=f"Model B: Actual vs Predicted (R²={r2_score(y_test, pred_b):.3f})",
             xlabel="Actual Profit", ylabel="Predicted Profit", xlim=(lo, hi), ylim=(lo, hi))
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

# (b) Residuals vs Predicted scatter
ax[0, 1].scatter(pred_b, resid, edgecolor="k", alpha=0.85)
ax[0, 1].axhline(0, color="r", ls="--")
ax[0, 1].set(title="Residuals vs Predicted", xlabel="Predicted Profit", ylabel="Residual")
ax[0, 1].grid(alpha=.3)

# (c) Q-Q plot
stats.probplot(resid, dist="norm", plot=ax[1, 0])
ax[1, 0].set_title("Residual Q-Q Plot"); ax[1, 0].grid(alpha=.3)

# (d) CV R² / Test R² 比較
plot_df = report.iloc[1:].set_index("Method")[["CV_R2", "Test_R2"]]
plot_df.plot.barh(ax=ax[1, 1])
ax[1, 1].set(title="CV R² vs Test R² by Method", xlabel="R²")
ax[1, 1].set_xlim(max(0, plot_df.min().min() - 0.05), 1.0)
ax[1, 1].grid(alpha=.3)

fig.tight_layout()
fig.savefig("main_plots.png", dpi=150)
print("Saved main_plots.png")

# 各特徵 vs Profit 散佈圖
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
fig2, axs = plt.subplots(1, 3, figsize=(16, 4.3))
for a, c in zip(axs, num_cols):
    a.scatter(df[c], df[TARGET], edgecolor="k", alpha=.8)
    m, b = np.polyfit(df[c], df[TARGET], 1)
    xs = np.linspace(df[c].min(), df[c].max(), 100)
    a.plot(xs, m * xs + b, "r-")
    a.set(xlabel=c, ylabel="Profit",
          title=f"{c} vs Profit (r={np.corrcoef(df[c], df[TARGET])[0, 1]:.2f})")
    a.grid(alpha=.3)
fig2.tight_layout()
fig2.savefig("main_feature_scatter.png", dpi=150)
print("Saved main_feature_scatter.png")


# ------------------------------------------------------------
# 8. TOP 10 OUTLIERS (全資料, Model B)
# ------------------------------------------------------------
section("8. TOP 10 OUTLIERS (Model B, full data)")
full_ols = sm.OLS(y, sm.add_constant(X)).fit()
infl = full_ols.get_influence()
out = pd.DataFrame({
    "Actual": y,
    "Predicted": full_ols.fittedvalues,
    "Residual": full_ols.resid,
    "Studentized_Resid": infl.resid_studentized_external,
    "Cooks_D": infl.cooks_distance[0],
    "Leverage": infl.hat_matrix_diag,
})
top10 = out.reindex(out["Studentized_Resid"].abs().sort_values(ascending=False).index).head(10)
print(top10.round(3).to_string())
top10.to_csv("top10_outliers.csv")
print("\n判斷: |Studentized Resid| > 3 通常視為離群值; Cook's D > 4/n 為高影響力點 "
      f"(4/n = {4 / len(y):.3f})")


# ------------------------------------------------------------
# 9. DEPLOYMENT
# ------------------------------------------------------------
section("9. DEPLOYMENT")
import joblib
final_cols = selected.get(best_name, features)
final = LinearRegression().fit(X[final_cols], y)
joblib.dump({"model": final, "features": final_cols}, "startup_mlr_final.joblib")
print(f"已儲存最終模型 ({best_name}), 特徵: {final_cols}")
