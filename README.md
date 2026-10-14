# NCHU AIoT - CRISP-DM 多元線性迴歸與特徵選擇整合專案

本專案實作 CRISP-DM (Cross-Industry Standard Process for Data Mining) 六大階段之多元線性迴歸 (Multiple Linear Regression) 流程，並整合 5 種特徵選擇演算法。

專案包含兩大資料集分析與一個可動態切換資料集的統一互動式探索網頁：
1. **50 Startups**: 透過 R&D、Administration、Marketing 支出與州別預測獲利 (Profit)。
2. **Boston Housing**: 透過 13 項社區與住宅特徵預測房價中位數 (MEDV)。

---

## 專案結構

```
NCHU_AIoT_1007_DIC/
├── explorer.html                  # 跨資料集統一互動式探索器 (支援下拉式選單切換資料集)
├── explorer_template_unified.html # 統一探索器 HTML 範本
├── .gitignore
├── README.md
├── 50_startups/                   # 50 Startups 專案目錄
│   ├── README.md
│   ├── data/
│   ├── explorer/                  # 獨立互動網頁 (50 Startups)
│   ├── figures/
│   ├── models/
│   ├── scripts/
│   └── tables/
└── boston_housing/                # Boston Housing 專案目錄
    ├── README.md
    ├── data/
    ├── explorer/                  # 獨立互動網頁 (Boston Housing)
    ├── figures/
    ├── models/
    ├── scripts/
    └── tables/
```

---

## 統一互動式探索器 (`explorer.html`)

根目錄下的 [explorer.html](file:///explorer.html) 整合了兩組資料集的特徵選擇探索工具：
- **下拉式選單**：即時在 **50 Startups (5 特徵)** 與 **Boston Housing (13 特徵)** 之間切換。
- **5 種特徵選擇演算法動態切換**：
  - Lasso (L1 Ranking)
  - RFECV (遞迴特徵消除)
  - Forward SFS (逐步向前選擇)
  - ElasticNetCV (L1 + L2 正則化)
  - Exhaustive Search (窮舉搜尋最佳子集)
- **即時指標與圖表**：
  - $R^2$、MAE、RMSE 隨特徵數量 $k$ 變化的動態折線圖
  - 特徵重要度水平長條圖
  - OLS 回歸係數與統計顯著性 ($p$-value) 摘要表
  - 殘差分佈與離群值檢視

---

## 快速開始

### 1. 安裝環境依賴

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels scipy joblib
```

### 2. 執行分析管線

#### 50 Startups:
```bash
python 50_startups/scripts/main.py
```

#### Boston Housing:
```bash
python boston_housing/scripts/main.py
python boston_housing/scripts/build_explorer.py
```

### 3. 開啟互動式探索網頁
直接用瀏覽器開啟專案根目錄下的 `explorer.html` 即可。
