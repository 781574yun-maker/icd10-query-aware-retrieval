# ============================================
# Hybrid_spacy.py( BERT + spaCy 分詞)
# ============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
import torch

import spacy   # <-- 使用 spaCy 中文分詞
nlp = spacy.load("zh_core_web_sm")

# spaCy 分詞函式
def spacy_cut(text):
    doc = nlp(text)
    return [t.text for t in doc]


# -------------------------------------------------
# 1) 讀取 ICD 詞庫
# -------------------------------------------------
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")

df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).str.strip()


# -------------------------------------------------
# 2) TF-IDF（用 spaCy 分詞）
# -------------------------------------------------
texts_cut = [" ".join(spacy_cut(t)) for t in df["text_for_search"]]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts_cut)


# -------------------------------------------------
# 3) BERT embedding
# -------------------------------------------------
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

corpus = df["text_for_search"].tolist()
bert_embeddings = bert_model.encode(corpus, convert_to_tensor=True)


# -------------------------------------------------
# 4) Hybrid 查詢功能
# -------------------------------------------------
def hybrid_search_spacy(query, top_k=5, alpha=0.5):
    """
    alpha = 1 → 完全 TF-IDF
    alpha = 0 → 完全 BERT
    """
    if not query.strip():
        return []

    # ===== TF-IDF =====
    query_cut = " ".join(spacy_cut(query))
    query_vec = vectorizer.transform([query_cut])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if tfidf_scores.max() > 0:
        tfidf_norm = tfidf_scores / tfidf_scores.max()
    else:
        tfidf_norm = tfidf_scores

    # ===== BERT =====
    q_emb = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.cos_sim(q_emb, bert_embeddings)[0].cpu().numpy()

    if bert_scores.max() > 0:
        bert_norm = bert_scores / bert_scores.max()
    else:
        bert_norm = bert_scores

    # ===== Hybrid =====
    hybrid_scores = alpha * tfidf_norm + (1 - alpha) * bert_norm

    top_idx = hybrid_scores.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "code": df.loc[i, "code_id"],
            "zh": df.loc[i, "zh_label"],
            "en": df.loc[i, "en_label"],
            "tfidf": float(tfidf_norm[i]),
            "bert": float(bert_norm[i]),
            "hybrid": float(hybrid_scores[i])
        })
    return results


# -------------------------------------------------
# 5) 測試互動查詢
# -------------------------------------------------
if __name__ == "__main__":
    print("Hybrid_spacy \n輸入 q 離開\n")

    while True:
        query = input("\n請輸入中文查詢詞：")
        if query.lower() == "q":
            print("bye!")
            break

        results = hybrid_search_spacy(query, top_k=5, alpha=0.6)

        if not results:
            print("（沒有結果）")
            continue

        print("\nHybrid 排序結果（Top-5）：\n")
        for r in results:
            print(f"{r['code']} | {r['zh']} ({r['en']}) "
                  f"| Hybrid：{r['hybrid']:.4f} | TF-IDF：{r['tfidf']:.4f} | BERT：{r['bert']:.4f}")
