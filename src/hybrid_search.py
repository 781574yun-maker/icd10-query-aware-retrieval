# ===========================================================
# Adaptive Hybrid v3  (左右 → jieba Hybrid / 長句 → spacy Hybrid / 其他 → BERT)
# 加入互動式查詢（跟你舊版格式完全相同）
# ===========================================================

import pandas as pd
import numpy as np
import jieba
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ===========================================================
# 0. spaCy 中文模型
# ===========================================================
nlp = spacy.load("zh_core_web_sm")

def spacy_cut(text):
    return [t.text for t in nlp(text)]

SYMPTOM = [
    "痛","脹","悶","暈","嘔吐","拉肚子","腹瀉","發燒",
    "咳嗽","呼吸困難","冒冷汗","癢","紅","腫"
]

# ===========================================================
# 1. 讀取 ICD 詞庫
# ===========================================================
print("載入 ICD 詞庫中...")
df = pd.read_csv("icd_vocab_new.csv").fillna("")
df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).astype(str)
corpus = df["text_for_search"].tolist()

# ===========================================================
# 2. 建立 TF-IDF（jieba / spaCy）
# ===========================================================
print("建立 jieba TF-IDF...")
jieba_corpus = [" ".join(jieba.cut(t)) for t in corpus]
vectorizer_jieba = TfidfVectorizer()
tfidf_jieba = vectorizer_jieba.fit_transform(jieba_corpus)

print("建立 spaCy TF-IDF...")
spacy_corpus = [" ".join(spacy_cut(t)) for t in corpus]
vectorizer_spacy = TfidfVectorizer()
tfidf_spacy = vectorizer_spacy.fit_transform(spacy_corpus)

# ===========================================================
# 3. BERT embeddings
# ===========================================================
print("載入 BERT 模型並建立 embeddings...")
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
bert_embeddings = bert_model.encode(corpus, convert_to_tensor=True)

# ===========================================================
# 4. 查詢特徵偵測
# ===========================================================
def detect_features(query):

    has_lateral = ("左" in query) or ("右" in query)

    has_separator = (
        "," in query or "，" in query or " " in query or "." in query or "。" in query
    )

    has_symptom = any(w in query for w in SYMPTOM)

    normal_long = (len(query) >= 10) and (not has_separator)

    return {
        "has_lateral": has_lateral,
        "has_separator": has_separator,
        "has_symptom": has_symptom,
        "normal_long": normal_long
    }

# ===========================================================
# 5. 決定 alpha（TF-IDF 佔比）
# ===========================================================
def decide_alpha(features):
    if features["has_lateral"]:
        return 0.7     # jieba + BERT
    if features["has_separator"]:
        return 0.5     # spacy + BERT
    if features["normal_long"]:
        return 0.0     # pure BERT
    if features["has_symptom"]:
        return 0.0     # pure BERT
    return 0.0         # other → pure BERT

# ===========================================================
# 6. 決定 TF-IDF 模式
# ===========================================================
def choose_tfidf(features):
    if features["has_lateral"]:
        return "jieba"
    if features["has_separator"]:
        return "spacy"
    return None   # pure BERT

# ===========================================================
# 7.  Hybrid Search
# ===========================================================
def hybrid_search(query, top_k=5):

    features = detect_features(query)
    alpha = decide_alpha(features)
    tfidf_mode = choose_tfidf(features)

    # --- TF-IDF ---
    if tfidf_mode == "jieba":
        q_cut = " ".join(jieba.cut(query))
        q_vec = vectorizer_jieba.transform([q_cut])
        tfidf_scores = cosine_similarity(q_vec, tfidf_jieba)[0]
        tfidf_norm = tfidf_scores / (tfidf_scores.max() + 1e-8)

    elif tfidf_mode == "spacy":
        q_cut = " ".join(spacy_cut(query))
        q_vec = vectorizer_spacy.transform([q_cut])
        tfidf_scores = cosine_similarity(q_vec, tfidf_spacy)[0]
        tfidf_norm = tfidf_scores / (tfidf_scores.max() + 1e-8)

    else:
        tfidf_norm = np.zeros(len(corpus))  # 不用 TF-IDF

    # --- BERT ---
    q_emb = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.cos_sim(q_emb, bert_embeddings)[0].cpu().numpy()
    bert_norm = bert_scores / (bert_scores.max() + 1e-8)

    # --- Hybrid ---
    hybrid_scores = alpha * tfidf_norm + (1 - alpha) * bert_norm

    top_idx = hybrid_scores.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "code": df.loc[i, "code_id"],
            "zh": df.loc[i, "zh_label"],
            "en": df.loc[i, "en_label"],
            "alpha": alpha,
            "tfidf_mode": tfidf_mode,
            "tfidf": float(tfidf_norm[i]),
            "bert": float(bert_norm[i]),
            "hybrid": float(hybrid_scores[i])
        })

    return results

# ===========================================================
# 8. 互動測試區（舊版格式完全相同）
# ===========================================================
if __name__ == "__main__":
    print("\n=== Adaptive Hybrid v3 查詢系統（輸入 q 離開） ===")

    while True:
        query = input("\n請輸入中文查詢詞：")
        if query.lower() == "q":
            print("bye!")
            break

        results = hybrid_search(query, top_k=5)

        if not results:
            print("（沒有結果）")
            continue

        print("\nHybrid 排序結果（Top-5）：\n")
        for r in results:
            print(f"{r['code']} | {r['zh']} ({r['en']})")
            print(f"Hybrid={r['hybrid']:.4f} | TF-IDF={r['tfidf']:.4f} | BERT={r['bert']:.4f} "
                  f"| α={r['alpha']} | 分詞={r['tfidf_mode']}")
            print("-" * 50)
