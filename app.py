"""
app.py
Unified Flask Web Server & REST API for CRISP-DM Multiple Linear Regression Explorer
Supports:
- 50 Startups (5 Features + Intercept)
- Boston Housing (13 Features + Intercept)
- Custom Dataset Analysis (/api/analyze)
"""

import os
import json
from flask import Flask, send_from_directory, request, jsonify
from scripts.analyze_dataset import analyze_dataset, load_data

app = Flask(__name__, static_folder=".", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load registries
DATASETS_REGISTRY = {}
for key, filename in [("startup", "model_data.json"), ("boston", "boston_model_data.json")]:
    p = os.path.join(BASE_DIR, filename)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                DATASETS_REGISTRY[key] = json.load(f)
        except Exception as e:
            print(f"Notice: Could not load {filename}: {e}")

@app.route("/")
def index():
    """Serve unified explorer dashboard."""
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/explorer.html")
def explorer():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/model_data.js")
def model_data_js():
    return send_from_directory(BASE_DIR, "model_data.js")

@app.route("/api/models", methods=["GET"])
def get_models():
    """Return model data registry for specified or default dataset."""
    dataset = request.args.get("dataset", "startup")
    algo = request.args.get("algo")
    reg = DATASETS_REGISTRY.get(dataset, DATASETS_REGISTRY.get("startup", {}))

    if algo and algo in reg:
        return jsonify({algo: reg[algo]})
    return jsonify(reg)

@app.route("/api/analyze", methods=["POST"])
def analyze_custom():
    """
    Accepts JSON with 'url' or 'csv_content', runs server-side MLR & feature selection,
    and returns full model specifications.
    """
    try:
        req_data = request.get_json(force=True) or {}
        url = req_data.get("url")
        target = req_data.get("target", "Profit")

        if not url:
            return jsonify({"status": "error", "message": "Missing 'url' parameter"}), 400

        df = load_data(url=url)
        results = analyze_dataset(df, target_col=target)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live prediction endpoint: Calculates target response based on active coefficients.
    """
    try:
        data = request.get_json(force=True) or {}
        dataset = data.get("dataset", "startup")
        algo = data.get("algo", "Lasso")
        k = int(data.get("k", 2))
        inputs = data.get("inputs", {})

        reg = DATASETS_REGISTRY.get(dataset, DATASETS_REGISTRY.get("startup", {}))
        if algo not in reg or k < 1 or k > len(reg[algo]):
            return jsonify({"error": "Invalid algorithm or k parameter"}), 400

        model_spec = reg[algo][k - 1]
        coefficients = model_spec["coefficients"]
        
        predicted_val = 0.0
        breakdown = []

        for c in coefficients:
            feat_name = c["feature"]
            coef_val = c["coef"]

            if "Intercept" in feat_name:
                predicted_val += coef_val
                breakdown.append({
                    "feature": "Intercept",
                    "coef": coef_val,
                    "val": 1.0,
                    "contribution": coef_val
                })
            else:
                x_val = float(inputs.get(feat_name, 0.0))
                contrib = coef_val * x_val
                predicted_val += contrib
                breakdown.append({
                    "feature": feat_name,
                    "coef": coef_val,
                    "val": x_val,
                    "contribution": contrib
                })

        return jsonify({
            "status": "success",
            "dataset": dataset,
            "algorithm": algo,
            "k": k,
            "predicted_value": predicted_val,
            "breakdown": breakdown
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" * CRISP-DM Unified MLR Explorer running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
