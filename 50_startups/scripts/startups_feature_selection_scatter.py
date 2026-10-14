"""
50 Startups: 比較多種特徵選擇演算法 + 散佈圖 (scatter plot)

用法: python startups_feature_selection_scatter.py [50_Startups.csv]
找不到 CSV 時,會自動產生欄位相同的模擬資料。
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 不開視窗,直接存圖;在本機想看視窗可刪掉這行
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, LassoCV, ElasticNetCV)
from sklearn.feature_selection import (SelectKBest, f_regression, RFE, RFECV,
                                       SequentialFeatureSelector, SelectFromModel)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

RANDOM_STATE = 42
TARGET = "Profit"


# ---------------------------------------------------------------------------
# 1. 讀資料 + One-Hot (State)
# ---------------------------------------------------------------------------
def load_data(path):
    if path and os.path.exists(path):
        print(f"Loaded {path}")
        return pd.read_csv(path)
    print("找不到 CSV -> 使用模擬資料")
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


df = load_data(sys.argv[1] if len(sys.argv) > 1 else "50_Startups.csv")
df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=float)

X = df.drop(columns=TARGET)
y = df[TARGET]
feature_names = np.array(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
Xtr = pd.DataFrame(scaler.transform(X_train), columns=feature_names, index=X_train.index)
Xte = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
lr = LinearRegression()

# ---------------------------------------------------------------------------
# 2. 各種特徵選擇演算法 (每個都回傳「被選中的欄位 mask」)
# ---------------------------------------------------------------------------
selectors = {
    "All features":      None,
    "SelectKBest (F-test, k=2)": SelectKBest(f_regression, k=2),
    "RFE (n=2)":         RFE(lr, n_features_to_select=2),
    "RFECV":             RFECV(lr, cv=cv, scoring="r2"),
    "Forward SFS (n=2)": SequentialFeatureSelector(lr, n_features_to_select=2,
                                                   direction="forward", cv=cv),
    "Backward SFS (n=2)": SequentialFeatureSelector(lr, n_features_to_select=2,
                                                    direction="backward", cv=cv),
    "Lasso (SelectFromModel)": SelectFromModel(LassoCV(cv=cv, max_iter=50_000,
                                                      random_state=RANDOM_STATE)),
    "ElasticNet (SelectFromModel)": SelectFromModel(
        ElasticNetCV(cv=cv, l1_ratio=[.2, .5, .8, 1], max_iter=50_000,
                     random_state=RANDOM_STATE)),
    "RandomForest importance": SelectFromModel(
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        threshold="mean"),
}

results, preds = [], {}
for name, sel in selectors.items():
    if sel is None:
        mask = np.ones(len(feature_names), dtype=bool)
    else:
        sel.fit(Xtr, y_train)
        mask = sel.get_support()
    if mask.sum() == 0:                      # 保險:至少留一個特徵
        mask[np.argmax(np.abs(np.corrcoef(Xtr.T, y_train)[-1, :-1]))] = True

    model = LinearRegression().fit(Xtr.loc[:, mask], y_train)
    pred = model.predict(Xte.loc[:, mask])
    preds[name] = pred
    results.append({
        "Method": name,
        "n_features": int(mask.sum()),
        "Selected": ", ".join(feature_names[mask]),
        "R2": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    })

report = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 200, "display.max_colwidth", 60)
print("\n=== 特徵選擇方法比較 (test set) ===")
print(report.round(3).to_string(index=False))
report.to_csv("feature_selection_results.csv", index=False)

# ---------------------------------------------------------------------------
# 3. 散佈圖 (scatter plot)
# ---------------------------------------------------------------------------
# 圖 1: 每個方法的 Actual vs Predicted
n = len(selectors)
cols = 3
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows))
axes = axes.ravel()
lo, hi = y_test.min() * 0.95, y_test.max() * 1.05

for ax, (name, pred) in zip(axes, preds.items()):
    r2 = r2_score(y_test, pred)
    ax.scatter(y_test, pred, alpha=0.8, edgecolor="k")
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_title(f"{name}\nR² = {r2:.3f}", fontsize=10)
    ax.set_xlabel("Actual Profit")
    ax.set_ylabel("Predicted Profit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
for ax in axes[n:]:
    ax.axis("off")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("50 Startups: Actual vs Predicted by Feature Selection Method", fontsize=14)
fig.tight_layout()
fig.savefig("scatter_actual_vs_predicted.png", dpi=150)
print("\nSaved scatter_actual_vs_predicted.png")

# 圖 2: 各特徵 vs Profit 的散佈圖 (含迴歸線)
num_cols = [c for c in X.columns if not c.startswith("State_")]
fig2, axes2 = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4.2))
axes2 = np.atleast_1d(axes2)
for ax, c in zip(axes2, num_cols):
    ax.scatter(df[c], df[TARGET], alpha=0.8, edgecolor="k")
    m, b = np.polyfit(df[c], df[TARGET], 1)
    xs = np.linspace(df[c].min(), df[c].max(), 100)
    ax.plot(xs, m * xs + b, "r-")
    ax.set_xlabel(c)
    ax.set_ylabel("Profit")
    ax.set_title(f"{c} vs Profit (r = {np.corrcoef(df[c], df[TARGET])[0, 1]:.2f})")
    ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("scatter_features_vs_profit.png", dpi=150)
print("Saved scatter_features_vs_profit.png")
