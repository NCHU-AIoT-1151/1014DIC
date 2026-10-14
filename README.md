# NCHU AIoT - CRISP-DM 多元線性迴歸與特徵選擇統一探索平台

本專案實作 CRISP-DM (Cross-Industry Standard Process for Data Mining) 六大階段之多元線性迴歸 (Multiple Linear Regression) 流程，整合 5 種特徵選擇演算法，並提供可動態輸入任意 Dataset URL 或上傳 CSV 的統一互動式探索平台。

UI 風格參考 [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)，具備現代化 SaaS 統計計算器美學、動態折線圖、迴歸方程式即時展示、參數檢定與 VIF 多重共線性診斷、即時模擬預測器以及殘差分析診斷。

---

## 🌟 核心特色

1. **跨資料集統一探索器 (`index.html` / `explorer.html`)**：
   - **50 Startups (5 特徵)**：以 R&D、Administration、Marketing 與州別預測 Profit，展示 $k=2$ 最佳精簡組合 (Occam's Razor) 與 $k=5$ 過度擬合峭壁。
   - **Boston Housing (13 特徵)**：以 13 項社區住宅特徵預測房價中位數 (MEDV)，展示 TAX 與 RAD 間的高多重共線性 (VIF > 26)。
2. **🌐 任意 Dataset 輸入與即時分析功能**：
   - 可直接輸入任何公開 CSV 網址（例如教授指定範例：`https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）。
   - 支援本機 `.csv` 檔案拖曳與上傳。
   - **瀏覽器端純前端高速運算引擎**：免開後端伺服器，直接在瀏覽器即時完成矩陣 OLS、80/20 切分、5 種特徵選擇演算法、ANOVA $F$-statistic、兩尾 $p$-value、95% 信賴區間、VIF 計算，並動態產生即時模擬滑桿與離群值殘差診斷表！
3. **5 種特徵選擇演算法動態切換**：
   - Lasso (L1 Ranking)
   - Forward SFS (逐步向前選擇)
   - RFE (遞迴特徵消除)
   - ElasticNet (L1 + L2 正則化)
   - Exhaustive Search (最佳子集窮舉搜尋)
   - ⚖️ **Compare All 5 Methods**：同軸對比 5 種演算法隨特徵數量 $k$ 變化的 RMSE 與 $R^2$ 走勢
4. **即時 What-If 預測模擬器**：
   - 自動依照當前選擇模型的活躍特徵生成對應數值範圍的滑動桿與輸入框，即時模擬預測結果與特徵貢獻度拆解。

---

## 專案結構

```
NCHU_AIoT_1014DIC/
├── index.html                     # 統一互動式探索平台 (權威入口，支援任意 URL/CSV 輸入)
├── explorer.html                  # 統一探索器鏡像 (向下相容)
├── model_data.js                  # 50 Startups & Boston Housing 預算完整模型資料庫
├── model_data.json                # 50 Startups 統計分析 JSON 註冊表
├── boston_model_data.json         # Boston Housing 統計分析 JSON 註冊表
├── app.py                         # Unified Flask 後端 (REST API & 靜態伺服器)
├── requirements.txt               # Python 依賴清單
├── scripts/
│   ├── analyze_dataset.py         # 任意 Dataset 命令列分析工具 (支援 --url 與 --file)
│   ├── build_unified_data.py      # 一鍵重算並產生所有模型資料的管線腳本
├── 50_startups/                   # 50 Startups 專案目錄
│   ├── data/50_Startups.csv       # 50 Startups 資料集
│   ├── explorer/explorer.html     # 統一導向入口
│   └── scripts/main.py            # 50 Startups 獨立分析腳本
└── boston_housing/                # Boston Housing 專案目錄
    ├── data/boston.csv            # Boston Housing 資料集
    ├── explorer/explorer.html     # 統一導向入口
    └── scripts/main.py            # Boston Housing 獨立分析腳本
```

---

## 快速開始

### 方式 A：直接以瀏覽器開啟（最快速）
直接雙擊開啟專案根目錄下的 [index.html](file:///index.html) 或 [explorer.html](file:///explorer.html) 即可！
- 在頂部下拉選單可自由切換 **50 Startups**、**Boston Housing** 或 **Custom Dataset**。
- 點選「**Input Dataset URL**」即可輸入任意 CSV 網址（預設填入 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`），點選「Fetch & Inspect」與「Run Multiple Linear Regression & Feature Selection」，即時輸出圖表與所有統計診斷！

---

### 方式 B：使用 Python Flask 本地伺服器

1. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

2. **啟動 Web 服務**：
   ```bash
   python app.py
   ```
   瀏覽器造訪：`http://localhost:5000/`

---

### 方式 C：使用命令列分析任意 Dataset

想要用 Python 自動分析任意 CSV 並匯出分析 JSON：
```bash
python scripts/analyze_dataset.py --url https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv --target Profit
```
或分析本機檔案：
```bash
python scripts/analyze_dataset.py --file path/to/your.csv --target your_target_col
```
輸出結果將直接儲存為結構化 JSON，可供網站或報告使用。

---

## 參考與致謝
- **介面設計參考**：[StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
- **資料集來源**：[Avik Jain 100-Days-Of-ML 50 Startups](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- **方法論**：CRISP-DM (Cross-Industry Standard Process for Data Mining)
