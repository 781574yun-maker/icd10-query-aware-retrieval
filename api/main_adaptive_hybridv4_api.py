# ============================================================
# main_adaptive_hybridv4_api.py
# Adaptive Hybrid v4 API（智慧模型選擇器）
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import joblib

# --- 讀取 3 個搜尋器 ---
from bert_model import bert_search
from Hybrid_Retrieval import hybrid_search as hybrid_jieba_search
from Hybrid_spacy import hybrid_search_spacy

# --- NLP feature extractor ---
from feature_extractor import extract_query_features

# ============================================================
# 1. 初始化 API
# ============================================================
app = FastAPI(title="ICD-10 Adaptive Hybrid v4 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. 載入 selector 模型 + label encoder
# ============================================================
SELECTOR_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\model_selector.pkl"
LABEL_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\label_encoder.pkl"

selector_model = joblib.load(SELECTOR_PATH)
label_encoder = joblib.load(LABEL_PATH)

print("Adaptive Selector Model Loaded!")


# ============================================================
# 3. 統一結果格式（沿用你的 normalize_results）
# ============================================================
def normalize_results(raw_results, model_name):

    normalized = []

    for r in raw_results:

        # --- BERT tuple 格式 ---
        if isinstance(r, tuple):
            code, name, score = r

            if "(" in name:
                zh = name.split("(")[0].strip()
                en = name.split("(")[1].replace(")", "").strip()
            else:
                zh = name
                en = ""

            normalized.append({
                "code": code,
                "zh": zh,
                "en": en,
                "score": float(score),
                "model": model_name
            })

        # --- Hybrid / Hybrid_spacy dict 格式 ---
        elif isinstance(r, dict):
            score = r.get("hybrid", r.get("score", 0))

            normalized.append({
                "code": r.get("code", ""),
                "zh": r.get("zh", ""),
                "en": r.get("en", ""),
                "score": float(score),
                "model": model_name
            })

    return normalized


# ============================================================
# 4. 核心查詢邏輯（完全照你的 adaptive_search）
# ============================================================
def adaptive_search(query: str, top_k=5):

    # ---- Step 1. NLP 特徵 ----
    feats = extract_query_features(query)

    feature_cols = [
        "char_len", "token_len_jieba", "token_len_spacy",
        "num_digits", "digit_ratio",
        "has_left_right", "has_body_part", "has_symptom",
        "has_postop", "has_multi_dx_sep",
        "has_punctuation", "is_long_sentence"
    ]

    X = pd.DataFrame([[feats[col] for col in feature_cols]], columns=feature_cols)

    # ---- Step 2. Selector 預測最佳模型 ----
    pred_label_id = selector_model.predict(X)[0]
    pred_model_name = label_encoder.inverse_transform([pred_label_id])[0]

    # ---- Step 3. 執行對應搜尋器 ----
    if pred_model_name == "bert":
        raw_results = bert_search(query, top_k=top_k)

    elif pred_model_name == "hybrid_jieba":
        raw_results = hybrid_jieba_search(query, top_k=top_k, alpha=0.6)

    elif pred_model_name == "hybrid_spacy":
        raw_results = hybrid_search_spacy(query, top_k=top_k, alpha=0.6)

    else:
        raw_results = []

    # ---- Step 4. 統一格式 ----
    final_results = normalize_results(raw_results, pred_model_name)

    # ---- Step 5. 回傳 JSON ----
    return {
        "query": query,
        "selected_model": pred_model_name,
        "results": final_results
    }


# ============================================================
# 5. API Request Body
# ============================================================
class QueryBody(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
# 6. /search 路由
# ============================================================
@app.post("/search")
def search_api(req: QueryBody):

    out = adaptive_search(req.query, req.top_k)

    return {
        "query": out["query"],
        "selected_model": out["selected_model"],
        "results": out["results"]
    }


# ============================================================
# 7. /model_info（企業最愛看）
# ============================================================
@app.get("/model_info")
def model_info():
    return {
        "model": "Adaptive Hybrid v4",
        "description": "智慧模型選擇器（自動判斷使用 BERT / Hybrid_jieba / Hybrid_spacy）",
        "features_used": [
            "char_len", "token_len_jieba", "token_len_spacy",
            "num_digits", "digit_ratio",
            "has_left_right", "has_body_part", "has_symptom",
            "has_postop", "has_multi_dx_sep",
            "has_punctuation", "is_long_sentence"
        ]
    }



#main_adaptive_hybridv4_api.py
#uvicorn main_adaptive_hybridv4_api:app --reload --port 8003
#http://127.0.0.1:8003/docs