# 50 Startups - CRISP-DM 多元線性迴歸

用 R&D / Administration / Marketing 支出與 State 預測新創公司 Profit,
比較 5 種特徵選擇方法, 找出最精簡的特徵組合 (parsimony)。

## 資料夾結構
```
50_startups/
├── data/        放 50_Startups.csv
├── scripts/     所有 Python 程式
├── explorer/    互動網頁 (產生器 + 範本 + 成品)
├── figures/     圖 (.png)
├── tables/      結果表 (.csv)
└── models/      存下來的模型 (.joblib)
```

## 腳本說明 (scripts/)
| 檔案 | 用途 |
|---|---|
| startups_crisp_dm.py | CRISP-DM 完整流程, 比較多種迴歸模型 |
| startups_multiple_linear_regression.py | 只做多元線性迴歸: p-value、backward elimination、殘差檢定 |
| startups_feature_selection_scatter.py | 9 種特徵選擇方法比較 + 散佈圖 |
| main.py | 教授風格主程式: Baseline、VIF、Model B、5 種特徵選擇、CV、圖、Top 10 離群值 |
| reproduce_figures.py | 圖 1 (RMSE/R² vs 特徵數) 與圖 2 (多模型比較) |
| all_5_algorithms_comparison_plot.py | 5 種方法在 k=1~5 的 RMSE / R² 比較圖 |

## 執行方式
先安裝: `pip install numpy pandas matplotlib scikit-learn statsmodels scipy joblib`
(圖 2 想含 LightGBM 再加 `lightgbm`)

腳本的輸出 (圖、csv、模型) 會存在「執行時所在的資料夾」, 建議這樣跑:
```
cd 50_startups
python scripts/main.py data/50_Startups.csv
python scripts/reproduce_figures.py data/50_Startups.csv
python scripts/all_5_algorithms_comparison_plot.py data/50_Startups.csv
```
跑完把新產生的檔案移到 figures/ tables/ models/ 歸檔即可。

互動網頁:
```
cd 50_startups/explorer
python build_explorer.py ../data/50_Startups.csv   # 產生 explorer.html, 用瀏覽器打開
```
(build_explorer.py 與 explorer_template.html 必須放在同一個資料夾)

## Workflow
Business Understanding → Data Understanding → Data Preparation (One-Hot、80/20 切分、標準化)
→ Baseline + VIF → Model B (5 特徵 + 截距) → 5 種特徵選擇
(LassoCV / RFECV / Forward SFS / ElasticNetCV / Exhaustive)
→ CV 與測試集評估 → 圖與殘差診斷 → Top 10 離群值 → 存模型 / 互動網頁

## 注意
- 目前 figures/ tables/ 內的數字來自模擬資料, 不是真實 50 Startups 的結果; 放入 CSV 重跑即可更新。
- 測試集只有 10 筆, 最佳特徵數會隨切分改變, 建議同時參考 CV 結果。
