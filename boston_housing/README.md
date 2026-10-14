# Boston Housing - CRISP-DM 多元線性迴歸

使用 13 個特徵 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) 預測房價中位數 (MEDV)。
依循 CRISP-DM 六大階段，比較 5 種特徵選擇方法 (Lasso L1 Ranking, RFECV, Forward SFS, ElasticNetCV, Exhaustive Search)，找出兼具預測力與精簡性 (parsimony) 的最佳特徵子集。

## 資料夾結構
```
boston_housing/
├── data/        放 boston.csv 與說明文件
├── scripts/     Python 程式 (main.py, build_explorer.py)
├── explorer/    互動式 Web 視覺化分析網頁 (explorer.html, explorer_template.html)
├── figures/     評估與診斷圖表 (.png)
├── tables/      模型比較表與離群值清單 (.csv)
└── models/      儲存之最佳模型 (.joblib)
```

## CRISP-DM 流程說明
1. **Business Understanding**: 定義預測目標 MEDV，建立評估指標 (Test $R^2 \ge 0.70$, MAE, RMSE)。
2. **Data Understanding**: 探索 13 個連續/數值變數與目標變數之相關性、描述統計與分佈。
3. **Data Preparation**: 80/20 切分 (random_state=42)，並以訓練集進行 StandardScaler 標準化以防止 Data Leakage。
4. **Baseline + VIF**: DummyRegressor 基準模型，並計算各特徵之 VIF 檢測共線性問題 (發現 RAD 與 TAX 存在強共線性)。
5. **Modeling & Feature Selection**: 
   - Model B (全特徵 OLS 回歸)
   - 5 種特徵選擇：LassoCV (L1)、RFECV、Forward SFS、ElasticNetCV、Exhaustive Search。
6. **Evaluation**: 5-Fold Cross Validation $R^2$、Test $R^2$ / MAE / RMSE、殘差常態性檢定 (Shapiro-Wilk) 與殘差診斷圖、Top 10 離群值分析。
7. **Deployment**: 模型持久化保存為 `.joblib`，並產出前端互動式視覺化分析工具。

## 執行方式
```bash
# 執行 CRISP-DM 完整迴歸管線
python boston_housing/scripts/main.py

# 產出 Boston Housing 獨立互動網頁
python boston_housing/scripts/build_explorer.py
```
