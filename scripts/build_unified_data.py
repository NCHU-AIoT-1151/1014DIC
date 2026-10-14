# scripts/build_unified_data.py
# Generates comprehensive model registries for 50 Startups and Boston Housing
import itertools, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, lars_path
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compute_vif_for_subset(df_X):
    if df_X.shape[1] <= 1:
        return {col: 1.0 for col in df_X.columns}
    vifs = {}
    cols = list(df_X.columns)
    for col in cols:
        other_cols = [c for c in cols if c != col]
        X_other = df_X[other_cols]
        y_curr = df_X[col]
        r2_curr = r2_score(y_curr, LinearRegression().fit(X_other, y_curr).predict(X_other))
        vif_val = 1.0 / (1.0 - r2_curr) if r2_curr < 0.99999 else 999.0
        vifs[col] = float(vif_val)
    return vifs

def build_model_data_for_dataset(df, target_col, categorical_cols=None, random_state=0, max_features=None):
    if categorical_cols:
        df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    else:
        df_enc = df.copy()

    X = df_enc.drop(columns=[target_col]).astype(float)
    y = df_enc[target_col].astype(float)
    features = list(X.columns)
    if max_features and len(features) > max_features:
        features = features[:max_features]
        X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)

    def evaluate_sub(sub):
        sub = list(sub)
        model = LinearRegression().fit(X_train[sub], y_train)
        pred = model.predict(X_test[sub])
        return np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)

    algorithms = {}
    K = len(features)

    # 1. Lasso
    try:
        lasso_cv = LassoCV(cv=5, random_state=random_state, max_iter=100000).fit(X_train_scaled, y_train)
        lasso_rank = [f for _, f in sorted(zip(abs(lasso_cv.coef_), features), reverse=True)]
    except Exception:
        lasso_rank = features[:]
    algorithms["Lasso"] = [lasso_rank[:k] for k in range(1, K + 1)]

    # 2. Forward SFS
    sfs_subsets = []
    for k in range(1, K + 1):
        if k == K:
            sfs_subsets.append(features[:])
        else:
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=3)
            sfs.fit(X_train, y_train)
            sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
    algorithms["Forward SFS"] = sfs_subsets

    # 3. RFE
    rfe_subsets = []
    for k in range(1, K + 1):
        if k == K:
            rfe_subsets.append(features[:])
        else:
            rfe = RFE(LinearRegression(), n_features_to_select=k)
            rfe.fit(X_train_scaled, y_train)
            rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
    algorithms["RFE"] = rfe_subsets

    # 4. ElasticNet
    try:
        enet_cv = ElasticNetCV(cv=5, random_state=random_state, max_iter=100000).fit(X_train_scaled, y_train)
        enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
    except Exception:
        enet_rank = features[:]
    algorithms["ElasticNet"] = [enet_rank[:k] for k in range(1, K + 1)]

    # 5. Exhaustive Search (up to min(K, 5) full search, beyond that Lasso rank)
    exhaustive_subsets = []
    for k in range(1, K + 1):
        if k <= 5:
            best_rmse = float("inf")
            best_sub = None
            for sub in itertools.combinations(features, k):
                rmse, _ = evaluate_sub(sub)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_sub = list(sub)
            exhaustive_subsets.append(best_sub)
        else:
            exhaustive_subsets.append(lasso_rank[:k])
    algorithms["Exhaustive"] = exhaustive_subsets

    output = {}
    for algo_name, subsets in algorithms.items():
        output[algo_name] = []
        for k_idx, sub in enumerate(subsets, start=1):
            model = LinearRegression().fit(X_train[sub], y_train)
            pred_test = model.predict(X_test[sub])
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
            mae = float(mean_absolute_error(y_test, pred_test))
            r2 = float(r2_score(y_test, pred_test))
            nt = len(y_test)
            adj_r2 = float(1 - (1 - r2) * (nt - 1) / max(nt - len(sub) - 1, 1))

            # statsmodels fit for p-values, t-stats, conf int
            X_tr_const = sm.add_constant(X_train[sub].astype(float), has_constant='add')
            ols_res = sm.OLS(y_train, X_tr_const).fit()
            f_stat = float(ols_res.fvalue) if not np.isnan(ols_res.fvalue) else 0.0
            f_pval = float(ols_res.f_pvalue) if not np.isnan(ols_res.f_pvalue) else 1.0

            vifs = compute_vif_for_subset(X_train[sub])

            coefficients = [{
                "feature": "Intercept (β₀)",
                "coef": float(ols_res.params['const']),
                "stderr": float(ols_res.bse['const']),
                "tstat": float(ols_res.tvalues['const']),
                "pvalue": float(ols_res.pvalues['const']),
                "ci_lower": float(ols_res.conf_int().loc['const', 0]),
                "ci_upper": float(ols_res.conf_int().loc['const', 1]),
                "vif": None
            }]

            for feat in sub:
                coefficients.append({
                    "feature": feat,
                    "coef": float(ols_res.params[feat]),
                    "stderr": float(ols_res.bse[feat]),
                    "tstat": float(ols_res.tvalues[feat]),
                    "pvalue": float(ols_res.pvalues[feat]),
                    "ci_lower": float(ols_res.conf_int().loc[feat, 0]),
                    "ci_upper": float(ols_res.conf_int().loc[feat, 1]),
                    "vif": round(vifs.get(feat, 1.0), 2)
                })

            # Outliers diagnostic on test set
            test_diffs = []
            for idx, (real, p_val) in enumerate(zip(y_test, pred_test)):
                orig_idx = y_test.index[idx]
                res = real - p_val
                abs_err = abs(res)
                pct_err = (abs_err / max(abs(real), 1e-5)) * 100
                test_diffs.append({
                    "index": int(orig_idx),
                    "actual": float(real),
                    "predicted": float(p_val),
                    "residual": float(res),
                    "abs_error": float(abs_err),
                    "pct_error": float(pct_err)
                })
            test_diffs.sort(key=lambda x: -x["abs_error"])
            for r_idx, item in enumerate(test_diffs[:10], start=1):
                item["rank"] = r_idx

            output[algo_name].append({
                "k": k_idx,
                "features": sub,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "adj_r2": adj_r2,
                "f_stat": f_stat,
                "f_pvalue": f_pval,
                "coefficients": coefficients,
                "outliers": test_diffs[:10]
            })

    return output

if __name__ == "__main__":
    print("Building 50 Startups data...")
    df_startups = pd.read_csv(os.path.join(BASE_DIR, "50_startups", "data", "50_Startups.csv"))
    startups_data = build_model_data_for_dataset(df_startups, target_col="Profit", categorical_cols=["State"], random_state=0)

    print("Building Boston Housing data...")
    df_boston = pd.read_csv(os.path.join(BASE_DIR, "boston_housing", "data", "boston.csv"))
    df_boston.columns = [c.upper() for c in df_boston.columns]
    boston_data = build_model_data_for_dataset(df_boston, target_col="MEDV", random_state=42)

    # Save to JSON
    with open(os.path.join(BASE_DIR, "model_data.json"), "w", encoding="utf-8") as f:
        json.dump(startups_data, f, indent=2)

    with open(os.path.join(BASE_DIR, "boston_model_data.json"), "w", encoding="utf-8") as f:
        json.dump(boston_data, f, indent=2)

    # Save combined JS
    js_content = f"""// Auto-generated unified dataset registry
window.DATASETS_REGISTRY = {{
  "startup": {json.dumps(startups_data)},
  "boston": {json.dumps(boston_data)}
}};
window.MODEL_DATA = window.DATASETS_REGISTRY["startup"];
"""
    with open(os.path.join(BASE_DIR, "model_data.js"), "w", encoding="utf-8") as f:
        f.write(js_content)

    print("All datasets generated and saved to model_data.js & JSON files!")
