# 產生互動網頁: python build_explorer.py [50_Startups.csv]  ->  explorer.html
import itertools, json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, lars_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RS, T = 42, "Profit"
path = sys.argv[1] if len(sys.argv) > 1 else "50_Startups.csv"
if os.path.exists(path):
    raw, src = pd.read_csv(path), "50_Startups.csv"
else:
    rng = np.random.default_rng(RS); n = 50
    raw = pd.DataFrame({"R&D Spend": rng.uniform(0, 165e3, n), "Administration": rng.uniform(51e3, 182e3, n),
                        "Marketing Spend": rng.uniform(0, 470e3, n),
                        "State": rng.choice(["New York", "California", "Florida"], n)})
    raw[T] = 5e4 + .85 * raw["R&D Spend"] + .03 * raw["Marketing Spend"] + rng.normal(0, 8e3, n)
    src = "synthetic demo data (same schema)"

df = pd.get_dummies(raw, columns=["State"], drop_first=True, dtype=float)
X, y = df.drop(columns=T), df[T]; F = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=RS)
sc = StandardScaler().fit(Xtr); Xs = pd.DataFrame(sc.transform(Xtr), columns=F, index=Xtr.index)
cv = KFold(5, shuffle=True, random_state=RS)
cvrmse = lambda c: -cross_val_score(LinearRegression(), Xtr[list(c)], ytr, cv=cv, scoring="neg_mean_squared_error").mean()

def order_to_subsets(o): return {k: [F[i] for i in o[:k]] for k in range(1, 6)}
_, act, _ = lars_path(Xs.values, ytr.values, method="lasso")
lo = list(dict.fromkeys(act)); lc = LassoCV(cv=cv, max_iter=100000, random_state=RS).fit(Xs, ytr)
lo += sorted(set(range(5)) - set(lo), key=lambda i: -abs(lc.coef_[i]))
en = ElasticNetCV(cv=cv, l1_ratio=[.1, .5, .9, 1], max_iter=100000, random_state=RS).fit(Xs, ytr)
S = {"Lasso (L1 Ranking)": order_to_subsets(lo)}
S["Forward SFS"] = {k: F[:] if k == 5 else [f for f, s in zip(F, SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=k, direction="forward", cv=cv,
    scoring="neg_mean_squared_error").fit(Xs, ytr).get_support()) if s] for k in range(1, 6)}
S["RFE"] = order_to_subsets(list(np.argsort(RFE(LinearRegression(), n_features_to_select=1).fit(Xs, ytr).ranking_)))
S["ElasticNet"] = order_to_subsets(list(np.argsort(-np.abs(en.coef_))))
S["Exhaustive Search"] = {k: list(min(itertools.combinations(F, k), key=cvrmse)) for k in range(1, 6)}

out = {"order": list(S), "src": src, "n_test": len(yte), "m": {}}
for m, d in S.items():
    out["m"][m] = {}
    for k, c in d.items():
        p = LinearRegression().fit(Xtr[c], ytr).predict(Xte[c]); r2 = r2_score(yte, p); nt = len(yte)
        o = sm.OLS(ytr, sm.add_constant(Xtr[c])).fit()
        out["m"][m][k] = dict(f=c, rmse=round(float(np.sqrt(mean_squared_error(yte, p))), 2), r2=round(r2, 4),
            adj=round(1 - (1 - r2) * (nt - 1) / (nt - k - 1), 4), mae=round(mean_absolute_error(yte, p), 2),
            F=round(float(o.fvalue), 1), p=float(o.f_pvalue))

HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explorer_template.html"), encoding="utf-8").read()
open("explorer.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(out)))
print("explorer.html written;", src)
