# ============================================
# eval_adaptive.py
# 計算 Adaptive Hybrid v4 的 Precision@K
# ============================================

import pandas as pd
from tqdm import tqdm
import joblib

# 匯入三個搜尋器
from bert_model import bert_search
from Hybrid_Retrieval import hybrid_search as hybrid_jieba_search
from Hybrid_spacy import hybrid_search_spacy

# NLP 特徵抽取
from feature_extractor import extract_query_features


# ============================================
# 1. 載入 Selector
# ============================================
SELECTOR_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\model_selector.pkl"
LABEL_PATH    = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\label_encoder.pkl"

selector_model = joblib.load(SELECTOR_PATH)
label_encoder  = joblib.load(LABEL_PATH)


# ============================================
# 2. Helper：統一格式（tuple/dict）
# ============================================
def normalize_results(raw_results):
    preds = []

    for r in raw_results:

        # BERT: (code, name, score)
        if isinstance(r, tuple):
            preds.append(str(r[0]).strip())

        # Hybrid: dict
        elif isinstance(r, dict):
            preds.append(str(r.get("code", "")).strip())

    return preds


# ============================================
# 3. Adaptive Search：套用 selector
# ============================================
def adaptive_predict_codes(query, top_k):

    # Step 1: 特徵
    feats = extract_query_features(query)

    feature_cols = [
        "char_len", "token_len_jieba", "token_len_spacy",
        "num_digits", "digit_ratio",
        "has_left_right", "has_body_part", "has_symptom",
        "has_postop", "has_multi_dx_sep",
        "has_punctuation", "is_long_sentence"
    ]
    X = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)

    # Step 2: selector 預測模型
    pred_label_id = selector_model.predict(X)[0]
    model_name = label_encoder.inverse_transform([pred_label_id])[0]

    # Step 3: 執行對應搜尋
    if model_name == "bert":
        raw_results = bert_search(query, top_k=top_k)

    elif model_name == "hybrid_jieba":
        raw_results = hybrid_jieba_search(query, top_k=top_k, alpha=0.6)

    elif model_name == "hybrid_spacy":
        raw_results = hybrid_search_spacy(query, top_k=top_k, alpha=0.6)

    else:
        raw_results = []

    # Step 4: 統一格式
    return normalize_results(raw_results)


# ============================================
# 4. Precision@K
# ============================================
def precision_at_k(gt_df, k):

    hits = 0
    total = len(gt_df)

    for _, row in tqdm(gt_df.iterrows(), total=total, desc=f"Adaptive@{k}"):

        query = row["diagnosis_text"]
        true_code = str(row["icd_code"]).strip()

        preds = adaptive_predict_codes(query, top_k=k)

        # 命中條件：前綴一致
        if any(p.startswith(true_code) for p in preds):
            hits += 1

    return hits / total


# ============================================
# 5. 主程式
# ============================================
if __name__ == "__main__":

    GT_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\clean_claim_gt.csv"
    gt_df = pd.read_csv(GT_PATH).fillna("")

    for k in [1, 5, 10]:
        p = precision_at_k(gt_df, k)
        print(f"Adaptive Hybrid v4 Precision@{k} = {p:.4f}")
