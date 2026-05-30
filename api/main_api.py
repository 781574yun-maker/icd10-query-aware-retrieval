# main_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from fastapi.middleware.cors import CORSMiddleware

# =======================================================
# 1) 初始化 FastAPI
# =======================================================
app = FastAPI(title="ICD-10-CM BERT Search API")

# 若你要給前端網頁用，CORS 必須打開（否則會被瀏覽器擋住）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 若你要限制來源，這裡可以改
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# 2) 載入 ICD 詞庫
# =======================================================
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")
df["text_for_search"] = (df["zh_label"].astype(str) + " " + df["en_label"].astype(str)).str.strip()

# =======================================================
# 3) 載入 BERT 模型（只會載入一次）
# =======================================================
print("載入 BERT 模型中...")
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
print("模型載入完成！")

# =======================================================
# 4) 預先建立 corpus embeddings（只做一次）
# =======================================================
print("編碼 ICD 詞庫中...")
corpus = df["text_for_search"].tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
print("嵌入建立完成！可開始查詢")

# =======================================================
# 5) 定義查詢輸入格式
# =======================================================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# =======================================================
# 6) 查詢函式（核心）
# =======================================================
def search_icd_by_bert(query: str, top_k: int = 5):
    query = query.strip()
    if not query:
        return []

    q_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, corpus_embeddings)[0]
    scores, indices = torch.topk(cos_scores, k=min(top_k, len(df)))

    results = []
    for s, idx in zip(scores, indices):
        row = df.iloc[int(idx)]
        results.append({
            "code_id": row["code_id"],
            "zh_label": row["zh_label"],
            "en_label": row["en_label"],
            "similarity": float(s.item())
        })
    return results

# =======================================================
# 7) FastAPI 路由 — /search
# =======================================================
@app.post("/search")
def search(request: QueryRequest):
    results = search_icd_by_bert(request.query, request.top_k)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": results
    }

# =======================================================
# 8) 啟動方式（用 uvicorn）
# =======================================================
#cd "C:\Users\User\Desktop\全球人壽產學\icd_project"
# 你在終端機執行：
# uvicorn main_api:app --reload --port 8000

#網頁
#http://127.0.0.1:8000/docs
#前端
# #fetch("http://127.0.0.1:8000/search", ...)
