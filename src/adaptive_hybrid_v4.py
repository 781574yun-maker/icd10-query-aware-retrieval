# ================================================
# adaptive_hybrid_v4.py（最終穩定版本）
# 完全相容你現有的三種搜尋器格式（tuple / dict）
# ================================================

import joblib
import pandas as pd

# --- 搜尋器 ---
from bert_model import bert_search
from Hybrid_Retrieval import hybrid_search as hybrid_jieba_search
from Hybrid_spacy import hybrid_search_spacy

# --- NLP 特徵抽取 ---
from feature_extractor import extract_query_features


# ==================================================
# 1. 載入 Selector 模型
# ==================================================
SELECTOR_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\model_selector.pkl"
LABEL_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\label_encoder.pkl"

selector_model = joblib.load(SELECTOR_PATH)
label_encoder = joblib.load(LABEL_PATH)

print("Selector Model Loaded")


# ==================================================
# 2. 統一格式轉換器（核心）
# ==================================================
def normalize_results(raw_results, model_name):
    """
    統一三種結果格式：
        1. BERT：[(code, "中文(英文)", score)]
        2. Hybrid：[{…}, …]
        3. Hybrid_spacy：[{…}, …]
    全部轉成同樣的 dict 形式。
    """
    normalized = []

    for r in raw_results:
        # --- BERT → tuple ---
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

        # --- Hybrid / Hybrid_spacy → dict ---
        elif isinstance(r, dict):
            score = r.get("hybrid", r.get("score", 0))

            normalized.append({
                "code": r.get("code", ""),
                "zh": r.get("zh", ""),
                "en": r.get("en", ""),
                "score": float(score),
                "model": model_name
            })

        # --- 不明格式（忽略） ---
        else:
            continue

    return normalized



# ==================================================
# 3. 主要函式：智慧查詢
# ==================================================
def adaptive_search(query: str, top_k=5):
    """
    1. 抽 NLP 特徵
    2. Selector 預測最佳模型
    3. 執行對應搜尋器
    4. 統一結果格式
    5. 回傳 JSON
    """

    # Step 1. NLP 特徵
    feats = extract_query_features(query)

    feature_cols = [
        "char_len", "token_len_jieba", "token_len_spacy",
        "num_digits", "digit_ratio",
        "has_left_right", "has_body_part", "has_symptom",
        "has_postop", "has_multi_dx_sep",
        "has_punctuation", "is_long_sentence"
    ]

    X = pd.DataFrame([[feats[col] for col in feature_cols]], columns=feature_cols)

    # Step 2. Selector 預測
    pred_label_id = selector_model.predict(X)[0]
    pred_model_name = label_encoder.inverse_transform([pred_label_id])[0]

    # Step 3. 執行對應搜尋器
    if pred_model_name == "bert":
        raw_results = bert_search(query, top_k=top_k)

    elif pred_model_name == "hybrid_jieba":
        raw_results = hybrid_jieba_search(query, top_k=top_k, alpha=0.6)

    elif pred_model_name == "hybrid_spacy":
        raw_results = hybrid_search_spacy(query, top_k=top_k, alpha=0.6)

    else:
        raw_results = []
        print("不明模型選擇")

    # Step 4. 統一格式
    final_results = normalize_results(raw_results, pred_model_name)

    # Step 5. 回傳 JSON
    return {
        "query": query,
        "selected_model": pred_model_name,
        "results": final_results
    }



# ==================================================
# 4. CLI 測試
# ==================================================
if __name__ == "__main__":
    print("\nAdaptive Hybrid v4（最終版）")
    print("輸入 q 離開\n")

    while True:
        query = input("請輸入查詢：")
        if query.lower() == "q":
            print("bye")
            break

        out = adaptive_search(query, top_k=5)

        print("\n=== 使用的模型 ===")
        print(out["selected_model"])

        print("\n=== Top-K 結果 ===")
        for r in out["results"]:
            print(f"{r['code']} | {r['zh']} | {r['en']} | score={r['score']:.4f}")
