# ============================================
# feature_extractor.py
# 功能：對 clean_claim_gt.csv 整批抽取 Query Features
# 產生 query_features.csv （給 Model Selector 訓練用）
# ============================================

import re
import pandas as pd
import jieba
import spacy

# --------------------------------------------
# 1. 你的檔案路徑（請確認）
# --------------------------------------------
GT_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\clean_claim_gt.csv"
OUT_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\query_features.csv"

# spaCy 中文模型
nlp = spacy.load("zh_core_web_sm")

# --------------------------------------------
# 2. 關鍵字字典（可隨時擴充）
# --------------------------------------------

LEFT_RIGHT_WORDS = ["左", "右", "右側", "左側", "上", "下", "前", "後"]
BODY_PART_WORDS = [
    "手", "腳", "臂", "腿", "肩", "膝", "腕", "指",
    "頭", "眼", "耳", "鼻", "口", "牙",
    "胸", "背", "腰", "腹", "股", "臀", "踝",
    "肘", "喉", "髖", "下顎", "臼齒"
]

SYMPTOM_WORDS = [
    "痛", "悶", "脹", "酸", "麻", "癢",
    "紅", "腫", "發燒", "嘔吐", "腹瀉", "拉肚子", "冒冷汗"
]

POSTOP_WORDS = ["術後", "手術後", "開刀後", "後遺"]
MULTI_DX_SEPARATORS = ["；", ";", "/", "、", "及", "，"]

PUNCT = ["，", ",", ".", "。", "；", ";", ":", "："]


# --------------------------------------------
# 3. 抽取單句特徵
# --------------------------------------------
def extract_query_features(q: str) -> dict:
    q = str(q).strip()

    # -------- 基本長度 --------
    char_len = len(q)

    # -------- jieba / spaCy tokens --------
    tokens_jieba = list(jieba.cut(q))
    tokens_spacy = [t.text for t in nlp(q)]

    token_len_jieba = len(tokens_jieba)
    token_len_spacy = len(tokens_spacy)

    # -------- 數字 --------
    digits = re.findall(r"\d", q)
    num_digits = len(digits)
    digit_ratio = num_digits / char_len if char_len > 0 else 0

    # -------- 語意型字詞檢查 --------
    has_left_right = any(w in q for w in LEFT_RIGHT_WORDS)
    has_body_part = any(w in q for w in BODY_PART_WORDS)
    has_symptom = any(w in q for w in SYMPTOM_WORDS)
    has_postop = any(w in q for w in POSTOP_WORDS)
    has_multi_dx_sep = any(sep in q for sep in MULTI_DX_SEPARATORS)

    # -------- 標點符號與長句判斷 --------
    has_punctuation = any(p in q for p in PUNCT)
    is_long_sentence = (char_len >= 15) or (token_len_spacy >= 10)

    # 回傳 feature dict
    return {
        "query": q,
        "char_len": char_len,
        "token_len_jieba": token_len_jieba,
        "token_len_spacy": token_len_spacy,
        "num_digits": num_digits,
        "digit_ratio": digit_ratio,
        "has_left_right": int(has_left_right),
        "has_body_part": int(has_body_part),
        "has_symptom": int(has_symptom),
        "has_postop": int(has_postop),
        "has_multi_dx_sep": int(has_multi_dx_sep),
        "has_punctuation": int(has_punctuation),
        "is_long_sentence": int(is_long_sentence),
    }


# --------------------------------------------
# 4. 批次處理（對 6 萬筆資料抽特徵）
# --------------------------------------------
def build_feature_table():
    print(f"Loading ground truth: {GT_PATH}")
    df = pd.read_csv(GT_PATH).fillna("")
    print(f"Total rows: {len(df)}")

    feature_rows = []

    for idx, row in df.iterrows():
        q = row["diagnosis_text"]
        feats = extract_query_features(q)

        # 加入 ground truth ICD
        feats["icd_code"] = row.get("icd_code", "")
        feats["row_id"] = idx  # primary key

        feature_rows.append(feats)

        # 顯示進度
        if (idx + 1) % 2000 == 0:
            print(f"Processed {idx+1} rows...")

    out = pd.DataFrame(feature_rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\nDone! Saved query_features.csv")
    print(f"Output shape: {out.shape}")


# --------------------------------------------
# 5. 主程式入口
# --------------------------------------------
if __name__ == "__main__":
    build_feature_table()
