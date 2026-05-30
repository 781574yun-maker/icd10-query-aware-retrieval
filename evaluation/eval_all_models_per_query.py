# ============================================
# eval_all_models_per_query.py (fixed version)
# 加入 startswith() → 解決 code 無法對應的問題
# ============================================

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import jieba
import spacy
import pickle
import os
from tqdm import tqdm

# --------------------------------------------
# 路徑設定
# --------------------------------------------
VOCAB_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
GT_PATH    = r"C:\Users\User\Desktop\全球人壽產學\icd_project\clean_claim_gt.csv"
MODEL_DIR  = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models"

OUT_PATH   = r"C:\Users\User\Desktop\全球人壽產學\icd_project\model_per_query.csv"

# --------------------------------------------
# 載入 ICD vocab
# --------------------------------------------
icd_df = pd.read_csv(VOCAB_PATH).fillna("")
icd_df["text_for_search"] = icd_df["zh_label"] + " " + icd_df["en_label"]
corpus = icd_df["text_for_search"].tolist()
code_list = icd_df["code_id"].tolist()

# --------------------------------------------
# 載入 ground truth
# --------------------------------------------
gt_df = pd.read_csv(GT_PATH)

# --------------------------------------------
# 載模型
# --------------------------------------------
def load_pkl(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

vec_jieba = load_pkl("tfidf_jieba_vectorizer.pkl")
mat_jieba = load_pkl("tfidf_jieba_matrix.pkl")

vec_spacy = load_pkl("tfidf_spacy_vectorizer.pkl")
mat_spacy = load_pkl("tfidf_spacy_matrix.pkl")

bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
emb_list = load_pkl("icd_embeddings.pkl")
corpus_embeddings = torch.from_numpy(np.vstack(emb_list))

nlp = spacy.load("zh_core_web_sm")

# --------------------------------------------
# 查詢函式
# --------------------------------------------
def bert_search(q, k):
    q_emb = bert_model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, corpus_embeddings)[0]
    idx = torch.topk(scores, k).indices.tolist()
    return [code_list[i] for i in idx]

def jieba_search(q, k):
    q_cut = " ".join(jieba.cut(q))
    q_vec = vec_jieba.transform([q_cut])
    scores = cosine_similarity(q_vec, mat_jieba)[0]
    idx = np.argsort(scores)[::-1][:k]
    return [code_list[i] for i in idx]

def spacy_search(q, k):
    q_cut = " ".join([t.text for t in nlp(q)])
    q_vec = vec_spacy.transform([q_cut])
    scores = cosine_similarity(q_vec, mat_spacy)[0]
    idx = np.argsort(scores)[::-1][:k]
    return [code_list[i] for i in idx]

# --------------------------------------------
# **最重要修正：ICD code 前段比對**
# --------------------------------------------
def icd_match(pred_list, true_code):
    return any(p.startswith(true_code) for p in pred_list)

# --------------------------------------------
# 主程式：跑所有模型的 per-query hit 結果
# --------------------------------------------
rows = []

for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df)):
    q = row["diagnosis_text"]
    true_code = str(row["icd_code"]).strip()

    record = {
        "row_id": idx,
        "query": q,
        "true_code": true_code,
    }

    # ---- BERT ----
    preds1 = bert_search(q, 1)
    preds5 = bert_search(q, 5)
    preds10 = bert_search(q, 10)

    record["bert_hit1"]  = int(icd_match(preds1, true_code))
    record["bert_hit5"]  = int(icd_match(preds5, true_code))
    record["bert_hit10"] = int(icd_match(preds10, true_code))

    # MRR (BERT)
    full_preds = bert_search(q, len(code_list))
    ranks = [i for i, p in enumerate(full_preds) if p.startswith(true_code)]
    record["bert_mrr"] = 1/(ranks[0]+1) if ranks else 0

    # ---- Hybrid Jieba ----
    preds_hj1 = jieba_search(q, 1)
    preds_hj5 = jieba_search(q, 5)
    preds_hj10 = jieba_search(q, 10)

    record["hybrid_jieba_hit1"]  = int(icd_match(preds_hj1, true_code))
    record["hybrid_jieba_hit5"]  = int(icd_match(preds_hj5, true_code))
    record["hybrid_jieba_hit10"] = int(icd_match(preds_hj10, true_code))

    # ---- Hybrid Spacy ----
    preds_hs1 = spacy_search(q, 1)
    preds_hs5 = spacy_search(q, 5)
    preds_hs10 = spacy_search(q, 10)

    record["hybrid_spacy_hit1"]  = int(icd_match(preds_hs1, true_code))
    record["hybrid_spacy_hit5"]  = int(icd_match(preds_hs5, true_code))
    record["hybrid_spacy_hit10"] = int(icd_match(preds_hs10, true_code))

    rows.append(record)

# --------------------------------------------
# 輸出
# --------------------------------------------
out = pd.DataFrame(rows)
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n✔ Done! Saved → {OUT_PATH}")
