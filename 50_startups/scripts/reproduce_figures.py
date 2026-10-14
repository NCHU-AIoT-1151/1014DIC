# ============================================================
# 重現兩種圖 (不修改 main.py, 沿用同樣的資料處理流程)
#   Figure 1: RMSE / R² by Number of Features  (+ 表格)
#   Figure 2: Comparison of Different Regression Models (+ 表格)
#
# pip install numpy pandas matplotlib scikit-learn
# (選用) pip install lightgbm
# ============================================================
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("(未安裝 lightgbm, 圖 2 會略過 LightGBM)")

RANDOM_STATE = 42
TARGET = "Profit"


# ------------------------------------------------------------
# 資料 (與 main.py 相同: One-Hot, drop_first)
# ------------------------------------------------------------
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
df = pd.get_dummies(raw, columns=["State"], drop_first=True, dtype=float)
X = df.drop(columns=TARGET)
y = df[TARGET]
features = list(X.columns)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)


def render_table(ax, table_df, fontsize=8):
    """把 DataFrame 以等寬字體畫在圖下方 (模擬 notebook 輸出)。"""
    ax.axis("off")
    ax.text(0.0, 1.0, table_df.to_string(index=False), family="monospace",
            fontsize=fontsize, va="top", ha="left", transform=ax.transAxes)


# ============================================================
# FIGURE 1: RMSE / R² by Number of Features
#   每個特徵數 k=1..5, 用 Exhaustive (itertools) 找 CV RMSE 最低的組合
# ============================================================
rows = []
for k in range(1, len(features) + 1):
    best = None
    for combo in itertools.combinations(features, k):
        mse = -cross_val_score(LinearRegression(), X[list(combo)], y, cv=cv,
                               scoring="neg_mean_squared_error").mean()
        if best is None or mse < best[0]:
            best = (mse, list(combo))
    r2 = cross_val_score(LinearRegression(), X[best[1]], y, cv=cv, scoring="r2").mean()
    rows.append({"Number of Features": k,
                 "Selected Features": "[" + ", ".join(best[1]) + "]",
                 "RMSE": np.sqrt(best[0]), "R-squared": r2})
fs = pd.DataFrame(rows)
fs.to_csv("fig1_feature_count_table.csv", index=False)
print("\n=== Figure 1 table (5-fold CV) ===")
print(fs.round(6).to_string(index=False))

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
a1, a2, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

for ax, col, title in [(a1, "RMSE", "RMSE by Number of Features"),
                       (a2, "R-squared", "R-squared by Number of Features")]:
    ax.plot(fs["Number of Features"], fs[col], marker="o")
    best_i = fs[col].idxmin() if col == "RMSE" else fs[col].idxmax()
    ax.scatter(fs.loc[best_i, "Number of Features"], fs.loc[best_i, col],
               s=140, facecolors="none", edgecolors="r", linewidths=2, label="Best")
    ax.set(title=title, xlabel="Number of Features", ylabel=col)
    ax.grid(alpha=.3); ax.legend()
render_table(at, fs.round(6), fontsize=8)
fig.tight_layout()
fig.savefig("fig1_feature_count.png", dpi=150)
print("Saved fig1_feature_count.png")


# ============================================================
# FIGURE 2: Comparison of Different Regression Models (單一特徵 R&D Spend)
#   曲線與 MSE/RMSE/R2 以全資料擬合 (in-sample) 計算; 另加 CV_R2 對照
# ============================================================
x1 = X[["R&D Spend"]]


def scaled(model):
    return Pipeline([("sc", StandardScaler()), ("m", model)])


def scaled_y(model):  # SVR 需要連 y 一起標準化
    return Pipeline([("sc", StandardScaler()),
                     ("m", TransformedTargetRegressor(regressor=model,
                                                      transformer=StandardScaler()))])


models = {
    "SVR": (scaled_y(SVR(kernel="rbf", C=10)), "navy"),
    "LASSO": (scaled(LassoCV(cv=cv, max_iter=100_000, random_state=RANDOM_STATE)), "green"),
    "Ridge": (scaled(Ridge(alpha=1.0)), "red"),
    "Elastic Net": (scaled(ElasticNetCV(cv=cv, l1_ratio=[.1, .5, .9, 1], max_iter=100_000,
                                        random_state=RANDOM_STATE)), "magenta"),
    "PCA": (Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=1)),
                      ("m", LinearRegression())]), "brown"),
    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE), "orange"),
    "Mutual Information": (Pipeline([("sel", SelectKBest(mutual_info_regression, k=1)),
                                     ("m", LinearRegression())]), "pink"),
    "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=RANDOM_STATE), "cyan"),
    "SVM": (scaled_y(SVR(kernel="linear", C=10)), "violet"),
}
if HAS_LGBM:
    models["LightGBM"] = (LGBMRegressor(random_state=RANDOM_STATE, verbose=-1), "gray")

grid = pd.DataFrame({"R&D Spend": np.linspace(x1.iloc[:, 0].min(), x1.iloc[:, 0].max(), 400)})

fig2 = plt.figure(figsize=(11, 11))
gs2 = fig2.add_gridspec(2, 1, height_ratios=[3, 1.2])
ax, at2 = fig2.add_subplot(gs2[0]), fig2.add_subplot(gs2[1])
ax.scatter(x1, y, color="black", label="Actual Data", zorder=5)

res = []
for name, (model, color) in models.items():
    model.fit(x1, y)
    pred = model.predict(x1)
    ax.plot(grid, model.predict(grid), color=color, lw=1.8, label=name)
    mse = mean_squared_error(y, pred)
    res.append({"Model": name, "Features": "[R&D Spend]" if name != "PCA" else "[PCA of R&D Spend]",
                "MSE": mse, "RMSE": np.sqrt(mse), "R2 Score": r2_score(y, pred),
                "CV_R2": cross_val_score(model, x1, y, cv=cv, scoring="r2").mean()})

ax.set(title="Comparison of Different Regression Models", xlabel="R&D Spend", ylabel="Profit")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.3)

tbl = pd.DataFrame(res).sort_values("MSE").reset_index(drop=True)
tbl.to_csv("fig2_model_comparison_table.csv", index=False)
print("\n=== Figure 2 table (MSE 由小到大) ===")
print(tbl.round(6).to_string(index=False))

show = tbl.copy()
show["MSE"] = show["MSE"].map("{:.6e}".format)
show[["RMSE", "R2 Score", "CV_R2"]] = show[["RMSE", "R2 Score", "CV_R2"]].round(6)
print()
at2.axis("off")
at2.text(0.0, 1.0, "Model results (sorted by MSE, ascending):\n" + show.to_string(index=False),
         family="monospace", fontsize=7.5, va="top", ha="left", transform=at2.transAxes)
fig2.tight_layout()
fig2.savefig("fig2_model_comparison.png", dpi=150)
print("Saved fig2_model_comparison.png")
