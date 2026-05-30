# ============================================
# Hybrid Retrieval (TF-IDF + BERT 加權混合)
# ============================================

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
import torch

# -------------------------------------------------
# 1) 讀取 ICD 詞庫
# -------------------------------------------------
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_en.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")

# 建立搜尋文本
df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).str.strip()

# -------------------------------------------------
# 2) 建立 TF-IDF（含 jieba 斷詞）
# -------------------------------------------------
texts_cut = [" ".join(jieba.cut(t)) for t in df["text_for_search"]]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts_cut)    # (N_docs, vocab_size)

# -------------------------------------------------
# 3) 載入 BERT + 預先建立 embedding
# -------------------------------------------------
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

corpus = df["text_for_search"].tolist()
bert_embeddings = bert_model.encode(corpus, convert_to_tensor=True)  # (N_docs, 512)

# -------------------------------------------------
# 4) Hybrid 查詢函式
# -------------------------------------------------
def hybrid_search(query, top_k=5, alpha=0.5):
    """
    alpha = 1 → 完全 TF-IDF
    alpha = 0 → 完全 BERT
    """
    query = query.strip()
    if not query:
        return []

    # ===== TF-IDF =====
    query_cut = " ".join(jieba.cut(query))
    query_vec = vectorizer.transform([query_cut])  # (1, vocab_size)
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 標準化（0~1）
    if tfidf_scores.max() > 0:
        tfidf_norm = tfidf_scores / tfidf_scores.max()
    else:
        tfidf_norm = tfidf_scores

    # ===== BERT =====
    q_emb = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.cos_sim(q_emb, bert_embeddings)[0].cpu().numpy()

    # 標準化（0~1）
    if bert_scores.max() > 0:
        bert_norm = bert_scores / bert_scores.max()
    else:
        bert_norm = bert_scores

    # ===== Hybrid Score =====
    hybrid_scores = alpha * tfidf_norm + (1 - alpha) * bert_norm

    # 取 Top-k
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
# 5) 互動式查詢
# -------------------------------------------------
if __name__ == "__main__":
    print("Hybrid Retrieval (TF-IDF + BERT)\n輸入 q 離開\n")

    while True:
        query = input("\n請輸入中文查詢詞：")
        if query.lower() == "q":
            print("bye!")
            break

        results = hybrid_search(query, top_k=5, alpha=0.6)

        if not results:
            print("（沒有結果）")
            continue

        print("\nHybrid 排序結果（Top-5）：\n")
        for r in results:
            print(f"{r['code']} | {r['zh']} ({r['en']}) "
                  f"| Hybrid：{r['hybrid']:.4f} | TF-IDF：{r['tfidf']:.4f} | BERT：{r['bert']:.4f}")
