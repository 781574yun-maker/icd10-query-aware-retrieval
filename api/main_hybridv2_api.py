#main_hybridv2_api.py
# ============================================
# Hybrid_spacy API (TF-IDF + BERT + spaCy)
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
import torch

import spacy  # spaCy 中文分詞
nlp = spacy.load("zh_core_web_sm")

# 分詞函式
def spacy_cut(text):
    return [t.text for t in nlp(text)]

# =======================================================
# 1. 初始化 FastAPI
# =======================================================
app = FastAPI(title="ICD-10 Hybrid (spaCy + BERT) API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# 2. 載入 ICD 詞庫
# =======================================================
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")
df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).str.strip()

# =======================================================
# 3. 建立 TF-IDF（spaCy 斷詞）
# =======================================================
print("建立 TF-IDF...")
texts_cut = [" ".join(spacy_cut(t)) for t in df["text_for_search"]]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts_cut)

# =======================================================
# 4. BERT embedding
# =======================================================
print("載入 BERT 模型中...")
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
print("編碼 ICD 詞庫中...")
corpus = df["text_for_search"].tolist()
bert_embeddings = bert_model.encode(corpus, convert_to_tensor=True)
print("BERT encoding 完成！")

# =======================================================
# 5. Request body
# =======================================================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5

# =======================================================
# 6. Hybrid 查詢邏輯（完全照你的版）
# =======================================================
def hybrid_search_spacy(query, top_k=5, alpha=0.5):
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

    # ===== Hybrid 加權 =====
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
            "hybrid": float(hybrid_scores[i]),
            "alpha_used": alpha
        })
    return results

# =======================================================
# 7. API 路由
# =======================================================
@app.post("/search")
def search(req: QueryRequest):
    results = hybrid_search_spacy(req.query, req.top_k, req.alpha)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "alpha": req.alpha,
        "results": results
    }

@app.get("/model_info")
def model_info():
    return {
        "model": "Hybrid Retrieval (TF-IDF + BERT + spaCy)",
        "description": "使用 spaCy 中文分詞 + TF-IDF + BERT 的混合語意搜尋模型"
    }



#main_hybridv2_api.py
#uvicorn main_hybridv2_api:app --reload --port 8002
#http://127.0.0.1:8002/docs
