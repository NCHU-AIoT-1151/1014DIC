# ============================================================
# 5 種特徵選擇方法比較: RMSE / R² vs Number of Features
# 沿用 main.py 的資料處理與演算法 (LassoCV, RFE, SFS, ElasticNetCV, Exhaustive)
# 輸出: all_5_algorithms_comparison_plot.png
#
# pip install numpy pandas matplotlib scikit-learn
# ============================================================
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 本機想跳出視窗就刪掉這行
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  lars_path)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

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
n_feat = len(features)

Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=features, index=X.index)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)


def cv_metrics(cols):
    """5 折 CV 的 RMSE 與 R² (OLS 不受縮放影響, 直接用原始特徵)。"""
    m = LinearRegression()
    mse = -cross_val_score(m, X[cols], y, cv=cv, scoring="neg_mean_squared_error").mean()
    r2 = cross_val_score(m, X[cols], y, cv=cv, scoring="r2").mean()
    return np.sqrt(mse), r2


# ------------------------------------------------------------
# 每種方法: 取得「特徵數 k -> 被選中的特徵」
# ------------------------------------------------------------
subsets = {}

# 1. Lasso (L1 Ranking): 依 Lasso 路徑中特徵「進入模型」的先後排序
_, active, _ = lars_path(Xs.values, y.values, method="lasso")
order = list(dict.fromkeys(active))                      # 去重並保留順序
lasso = LassoCV(cv=cv, max_iter=100_000, random_state=RANDOM_STATE).fit(Xs, y)
rest = sorted(set(range(n_feat)) - set(order), key=lambda i: -abs(lasso.coef_[i]))
order += rest
subsets["Lasso (L1 Ranking)"] = {k: [features[i] for i in order[:k]] for k in range(1, n_feat + 1)}

# 2. Forward SFS: 每個 k 各跑一次前向選擇
subsets["Forward SFS"] = {}
for k in range(1, n_feat + 1):
    if k == n_feat:
        subsets["Forward SFS"][k] = features[:]
        continue
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k,
                                    direction="forward", cv=cv,
                                    scoring="neg_mean_squared_error").fit(Xs, y)
    subsets["Forward SFS"][k] = [f for f, s in zip(features, sfs.get_support()) if s]

# 3. RFE: ranking_ = 1 表示最後被留下 (最重要)
rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y)
rfe_order = [features[i] for i in np.argsort(rfe.ranking_)]
subsets["RFE"] = {k: rfe_order[:k] for k in range(1, n_feat + 1)}

# 4. ElasticNet (Ranking): 依 |係數| 由大到小
enet = ElasticNetCV(cv=cv, l1_ratio=[.1, .3, .5, .7, .9, 1], max_iter=100_000,
                    random_state=RANDOM_STATE).fit(Xs, y)
en_order = [features[i] for i in np.argsort(-np.abs(enet.coef_))]
subsets["ElasticNet Ranking"] = {k: en_order[:k] for k in range(1, n_feat + 1)}

# 5. Exhaustive Search: 每個 k 試完所有組合, 取 CV RMSE 最低
subsets["Exhaustive Search"] = {}
for k in range(1, n_feat + 1):
    best = min(itertools.combinations(features, k), key=lambda c: cv_metrics(list(c))[0])
    subsets["Exhaustive Search"][k] = list(best)

# ------------------------------------------------------------
# 評估
# ------------------------------------------------------------
rows = []
for method, d in subsets.items():
    for k, cols in d.items():
        rmse, r2 = cv_metrics(cols)
        rows.append({"Method": method, "Number of Features": k,
                     "Selected Features": ", ".join(cols), "RMSE": rmse, "R-squared": r2})
res = pd.DataFrame(rows)
res.to_csv("all_5_algorithms_results.csv", index=False)

pd.set_option("display.width", 220, "display.max_colwidth", 90)
print("\n=== 各方法各特徵數的結果 (5-fold CV) ===")
print(res.round(4).to_string(index=False))

# ------------------------------------------------------------
# 圖: RMSE / R² Comparison across 5 Feature Selection Methods
# ------------------------------------------------------------
styles = {
    "Lasso (L1 Ranking)": ("o", "tab:blue"),
    "Forward SFS": ("s", "tab:orange"),
    "RFE": ("^", "tab:green"),
    "ElasticNet Ranking": ("D", "tab:red"),
    "Exhaustive Search": ("v", "tab:purple"),
}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, col, ylabel, title, loc in [
    (axes[0], "RMSE", "RMSE (Lower is Better)",
     "RMSE Comparison across 5 Feature Selection Methods", "upper right"),
    (axes[1], "R-squared", "R-squared (Higher is Better)",
     "R-squared Comparison across 5 Feature Selection Methods", "lower left"),
]:
    for method, (marker, color) in styles.items():
        sub = res[res["Method"] == method]
        ax.plot(sub["Number of Features"], sub[col], marker=marker, color=color,
                label=method, lw=1.8, markersize=6)
    ax.set_xticks(range(1, n_feat + 1))
    ax.set(title=title, xlabel="Number of Features", ylabel=ylabel)
    ax.grid(alpha=.3, ls="--")
    ax.legend(loc=loc, fontsize=9)
fig.tight_layout()
fig.savefig("all_5_algorithms_comparison_plot.png", dpi=150)
print("\nSaved all_5_algorithms_comparison_plot.png")
