#main_hybridv1_api.py
# ============================================
# Hybrid Retrieval API (jieba + TF-IDF + BERT)
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
import torch

# =======================================================
# 1. 初始化 API
# =======================================================
app = FastAPI(title="ICD-10 Hybrid (jieba + BERT) API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# 2. 讀取 ICD 詞庫
# =======================================================
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")

df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).str.strip()

# =======================================================
# 3. 建立 TF-IDF + jieba
# =======================================================
texts_cut = [" ".join(jieba.cut(t)) for t in df["text_for_search"]]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts_cut)

# =======================================================
# 4. 載入 BERT + 預建 embedding
# =======================================================
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
corpus = df["text_for_search"].tolist()
bert_embeddings = bert_model.encode(corpus, convert_to_tensor=True)

# =======================================================
# 5. 定義 Request Body
# =======================================================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5  # 預設：TF-IDF 與 BERT 各 0.5 權重

# =======================================================
# 6. Hybrid 查詢（沿用你原本的邏輯）
# =======================================================
def hybrid_search(query, top_k=5, alpha=0.5):

    query = query.strip()
    if not query:
        return []

    # ===== TF-IDF =====
    query_cut = " ".join(jieba.cut(query))
    query_vec = vectorizer.transform([query_cut])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 標準化
    if tfidf_scores.max() > 0:
        tfidf_norm = tfidf_scores / tfidf_scores.max()
    else:
        tfidf_norm = tfidf_scores

    # ===== BERT =====
    q_emb = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.cos_sim(q_emb, bert_embeddings)[0].cpu().numpy()

    # 標準化
    if bert_scores.max() > 0:
        bert_norm = bert_scores / bert_scores.max()
    else:
        bert_norm = bert_scores

    # ===== Hybrid 加權 =====
    hybrid_scores = alpha * tfidf_norm + (1 - alpha) * bert_norm

    # 排序取前 k
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
            "alpha_used": alpha,
        })
    return results

# =======================================================
# 7. API 路由 — /search
# =======================================================
@app.post("/search")
def search(req: QueryRequest):
    results = hybrid_search(req.query, req.top_k, req.alpha)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "alpha": req.alpha,
        "results": results
    }

# =======================================================
# 8. 模型資訊（給前端或企業）
# =======================================================
@app.get("/model_info")
def model_info():
    return {
        "model": "Hybrid Retrieval (TF-IDF + BERT + jieba)",
        "description": "混合語意與文字匹配，可調整 alpha 作為 TF-IDF/BERT 權重"
    }


#main_hybridv1_api.py
#uvicorn main_hybridv1_api:app --reload --port 8001
#http://127.0.0.1:8001/docs
