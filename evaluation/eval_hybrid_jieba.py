# eval_hybrid_jieba.py — Hybrid (jieba + BERT)
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import jieba
from tqdm import tqdm
import pickle
import os

VOCAB_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
GT_PATH    = r"C:\Users\User\Desktop\全球人壽產學\icd_project\clean_claim_gt.csv"
MODEL_DIR  = r"C:\Users\User\Desktop\全球人壽產學\icd_project\models"

print(">>> Loading ICD vocab…")
icd_df = pd.read_csv(VOCAB_PATH).fillna("")
icd_df["text_for_search"] = icd_df["zh_label"] + " " + icd_df["en_label"]
corpus = icd_df["text_for_search"].tolist()
code_list = icd_df["code_id"].tolist()

print(">>> Loading Ground Truth…")
gt_df = pd.read_csv(GT_PATH)

print(">>> Loading TF-IDF + BERT…")
with open(os.path.join(MODEL_DIR, "tfidf_jieba_vectorizer.pkl"), "rb") as f:
    vec_j = pickle.load(f)
with open(os.path.join(MODEL_DIR, "tfidf_jieba_matrix.pkl"), "rb") as f:
    mat_j = pickle.load(f)

bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
with open(os.path.join(MODEL_DIR, "icd_embeddings.pkl"), "rb") as f:
    emb_list = pickle.load(f)
corpus_embeddings = torch.from_numpy(np.vstack(emb_list))

def hybrid_jieba(q, k):
    q_cut = " ".join(jieba.cut(q))
    tfidf_vec = vec_j.transform([q_cut])
    tfidf_scores = cosine_similarity(tfidf_vec, mat_j)[0]
    tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)

    q_emb = bert_model.encode(q, convert_to_tensor=True)
    bert_scores = util.cos_sim(q_emb, corpus_embeddings)[0].cpu().numpy()
    bert_norm = (bert_scores - bert_scores.min()) / (bert_scores.max() - bert_scores.min() + 1e-8)

    hybrid = 0.5 * tfidf_norm + 0.5 * bert_norm
    idx = np.argsort(hybrid)[::-1][:k]
    return [code_list[i] for i in idx]

def precision_at_k(model_fn, k):
    hits = 0
    total = len(gt_df)
    for _, row in tqdm(gt_df.iterrows(), total=total):
        q = row["diagnosis_text"]
        true_code = str(row["icd_code"]).strip()

        preds = model_fn(q, k)
        if any(p.startswith(true_code) for p in preds):
            hits += 1

    return hits / total

ks = [1, 5, 10]
for k in ks:
    p = precision_at_k(hybrid_jieba, k)
    print(f"Hybrid (jieba) Precision@{k} = {p:.4f}")
