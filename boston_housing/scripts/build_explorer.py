# Generate interactive HTML explorer: python build_explorer.py [boston.csv] -> explorer.html
import itertools, json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, lars_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RS, T = 42, "MEDV"
MAX_K_EXHAUSTIVE = 6   # limit exhaustive search to k<=6 (C(13,6)=1716 combos)
BOSTON_EXPLORER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "explorer")

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "boston.csv")

if os.path.exists(path):
    raw = pd.read_csv(path)
    if raw.columns[0].lower() in ("", "unnamed: 0"):
        raw = raw.iloc[:, 1:]
    raw.columns = [c.upper() for c in raw.columns]
    src = os.path.basename(path)
else:
    rng = np.random.default_rng(RS); n = 506
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
    raw = pd.DataFrame({
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas,
        "NOX": nox, "RM": rm, "AGE": age, "DIS": dis,
        "RAD": rad, "TAX": tax, "PTRATIO": ptrat, "B": b,
        "LSTAT": lstat, T: medv,
    })
    src = "synthetic demo data (Boston Housing schema)"

df = raw.dropna().copy()
X, y = df.drop(columns=T), df[T]
F = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=RS)
sc = StandardScaler().fit(Xtr)
Xs = pd.DataFrame(sc.transform(Xtr), columns=F, index=Xtr.index)
cv = KFold(5, shuffle=True, random_state=RS)
cvrmse = lambda c: float(-cross_val_score(LinearRegression(), Xtr[list(c)], ytr, cv=cv,
                                          scoring="neg_mean_squared_error").mean())

def order_to_subsets(o, max_k=13):
    return {k: [F[i] for i in o[:k]] for k in range(1, max_k+1)}

_, act, _ = lars_path(Xs.values, ytr.values, method="lasso")
lo_idx = list(dict.fromkeys(act))
lc = LassoCV(cv=cv, max_iter=200000, random_state=RS).fit(Xs, ytr)
missing = sorted(set(range(len(F))) - set(lo_idx), key=lambda i: -abs(lc.coef_[i]))
lo_idx += missing
S = {"Lasso (L1 Ranking)": order_to_subsets(lo_idx)}

S["Forward SFS"] = {}
for k in range(1, len(F)+1):
    if k == len(F):
        S["Forward SFS"][k] = F[:]
    else:
        sel = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=k, direction="forward",
            cv=cv, scoring="neg_mean_squared_error").fit(Xs, ytr)
        S["Forward SFS"][k] = [f for f, s in zip(F, sel.get_support()) if s]

rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, ytr)
S["RFE"] = order_to_subsets(list(np.argsort(rfe.ranking_)))

en = ElasticNetCV(cv=cv, l1_ratio=[.1, .5, .9, 1], max_iter=200000, random_state=RS).fit(Xs, ytr)
S["ElasticNet"] = order_to_subsets(list(np.argsort(-np.abs(en.coef_))))

# Exhaustive search limited to MAX_K_EXHAUSTIVE; beyond that use Lasso order
print(f"Exhaustive search up to k={MAX_K_EXHAUSTIVE} ...")
S["Exhaustive Search"] = {}
for k in range(1, len(F)+1):
    if k <= MAX_K_EXHAUSTIVE:
        S["Exhaustive Search"][k] = list(min(itertools.combinations(F, k), key=cvrmse))
    else:
        S["Exhaustive Search"][k] = S["Lasso (L1 Ranking)"][k]  # fall back to Lasso feature names

out = {"order": list(S), "src": src, "n_test": len(yte), "n_features_total": len(F), "m": {}}
for m, d in S.items():
    out["m"][m] = {}
    for k, c in d.items():
        p = LinearRegression().fit(Xtr[c], ytr).predict(Xte[c])
        r2 = r2_score(yte, p); nt = len(yte)
        o = sm.OLS(ytr, sm.add_constant(Xtr[c].astype(float))).fit()
        out["m"][m][k] = dict(
            f=c,
            rmse=round(float(np.sqrt(mean_squared_error(yte, p))), 4),
            r2=round(float(r2), 4),
            adj=round(float(1 - (1-r2)*(nt-1)/(nt-k-1)), 4),
            mae=round(float(mean_absolute_error(yte, p)), 4),
            F=round(float(o.fvalue), 2),
            p=float(o.f_pvalue))


tmpl_path = os.path.join(BOSTON_EXPLORER, "explorer_template.html")
out_path  = os.path.join(BOSTON_EXPLORER, "explorer.html")
os.makedirs(BOSTON_EXPLORER, exist_ok=True)
if os.path.exists(tmpl_path):
    html = open(tmpl_path, encoding="utf-8").read()
    open(out_path, "w", encoding="utf-8").write(html.replace("__DATA__", json.dumps(out)))
    print(f"explorer.html written to: {out_path}")
    print(f"Data source: {src}")
else:
    print(f"Template not found at {tmpl_path}; run write_html_files.py first.")
    with open(os.path.join(BOSTON_EXPLORER, "boston_data.json"), "w") as jf:
        json.dump(out, jf)
    print(f"Data saved to boston_data.json")
