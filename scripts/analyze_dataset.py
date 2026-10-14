"""
scripts/analyze_dataset.py
Automated Multiple Linear Regression (MLR) & 5 Feature Selection Algorithms Analyzer
Accepts any CSV file or direct URL, executes full CRISP-DM regression pipeline,
computes inferential statistics (OLS, t-test, p-value, 95% CI, VIF),
and outputs structured JSON for web visualization.
"""

import argparse
import io
import itertools
import json
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def compute_vif_for_subset(df_X):
    if df_X.shape[1] <= 1:
        return {col: 1.0 for col in df_X.columns}
    vifs = {}
    cols = list(df_X.columns)
    for col in cols:
        other_cols = [c for c in cols if c != col]
        X_other = df_X[other_cols]
        y_curr = df_X[col]
        try:
            r2_curr = r2_score(y_curr, LinearRegression().fit(X_other, y_curr).predict(X_other))
            vif_val = 1.0 / (1.0 - r2_curr) if r2_curr < 0.99999 else 999.0
        except Exception:
            vif_val = 1.0
        vifs[col] = float(vif_val)
    return vifs

def analyze_dataset(df, target_col, categorical_cols=None, random_state=0, max_k=15):
    # Auto-detect target column if not found
    if target_col not in df.columns:
        # Case insensitive match
        matches = [c for c in df.columns if c.lower() == target_col.lower()]
        if matches:
            target_col = matches[0]
        else:
            # pick last numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                raise ValueError("No numeric columns found for target variable.")
            target_col = num_cols[-1]
            print(f"Target '{target_col}' auto-selected as default continuous response.")

    # Drop unneeded identifier columns if any
    for id_col in ["id", "ID", "Unnamed: 0", "Unnamed: 0.1"]:
        if id_col in df.columns and id_col != target_col:
            df = df.drop(columns=[id_col])

    # Auto-detect categorical columns if not explicitly provided
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if df[c].dtype == object and c != target_col]

    if categorical_cols:
        df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    else:
        df_enc = df.copy()

    df_enc = df_enc.dropna()
    X = df_enc.drop(columns=[target_col]).astype(float)
    y = df_enc[target_col].astype(float)
    features = list(X.columns)

    if max_k and len(features) > max_k:
        features = features[:max_k]
        X = X[features]

    K = len(features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)

    def evaluate_sub(sub):
        sub = list(sub)
        model = LinearRegression().fit(X_train[sub], y_train)
        pred = model.predict(X_test[sub])
        return np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)

    algorithms = {}

    # 1. Lasso
    try:
        lasso_cv = LassoCV(cv=min(5, len(X_train)), random_state=random_state, max_iter=100000).fit(X_train_scaled, y_train)
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
            try:
                sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=min(3, len(X_train)))
                sfs.fit(X_train, y_train)
                sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
            except Exception:
                sfs_subsets.append(lasso_rank[:k])
    algorithms["Forward SFS"] = sfs_subsets

    # 3. RFE
    rfe_subsets = []
    for k in range(1, K + 1):
        if k == K:
            rfe_subsets.append(features[:])
        else:
            try:
                rfe = RFE(LinearRegression(), n_features_to_select=k)
                rfe.fit(X_train_scaled, y_train)
                rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
            except Exception:
                rfe_subsets.append(lasso_rank[:k])
    algorithms["RFE"] = rfe_subsets

    # 4. ElasticNet
    try:
        enet_cv = ElasticNetCV(cv=min(5, len(X_train)), random_state=random_state, max_iter=100000).fit(X_train_scaled, y_train)
        enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
    except Exception:
        enet_rank = features[:]
    algorithms["ElasticNet"] = [enet_rank[:k] for k in range(1, K + 1)]

    # 5. Exhaustive Search
    exhaustive_subsets = []
    for k in range(1, K + 1):
        if k <= 5 and (K <= 10 or k <= 4):
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

    meta = {
        "dataset_name": getattr(df, "name", "Custom Dataset"),
        "target": target_col,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(features),
        "all_features": features,
        "feature_ranges": {
            feat: {
                "min": float(X[feat].min()),
                "max": float(X[feat].max()),
                "mean": float(X[feat].mean()),
                "median": float(X[feat].median())
            } for feat in features
        }
    }

    return {"metadata": meta, "models": output}

def load_data(url=None, file_path=None):
    if url:
        print(f"Fetching dataset from URL: {url} ...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        content = urllib.request.urlopen(req).read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        df.name = os.path.basename(url.split("?")[0]) or "Remote_CSV"
        return df
    elif file_path:
        print(f"Loading local dataset: {file_path} ...")
        df = pd.read_csv(file_path)
        df.name = os.path.basename(file_path)
        return df
    else:
        raise ValueError("Must provide either --url or --file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze any dataset with Multiple Linear Regression & 5 Feature Selection Algorithms.")
    parser.add_argument("--url", help="Direct URL to CSV dataset (e.g. raw.githubusercontent.com)")
    parser.add_argument("--file", help="Local file path to CSV dataset")
    parser.add_argument("--target", default="Profit", help="Target column name (e.g. Profit, MEDV, price)")
    parser.add_argument("--output", default="custom_model_data.json", help="Output path for JSON results")
    args = parser.parse_args()

    target_url = args.url or ("https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv" if not args.file else None)
    df = load_data(url=target_url, file_path=args.file)
    print(f"Dataset Loaded: Shape = {df.shape}, Columns = {list(df.columns)}")

    results = analyze_dataset(df, target_col=args.target)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete! Saved structured analysis to: {args.output}")
