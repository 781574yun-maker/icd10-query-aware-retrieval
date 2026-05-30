# eval_bert.py — 單核心 BERT Precision@k
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
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

print(">>> Loading BERT model + embeddings…")
bert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

with open(os.path.join(MODEL_DIR, "icd_embeddings.pkl"), "rb") as f:
    emb_list = pickle.load(f)
corpus_embeddings = torch.from_numpy(np.vstack(emb_list))

def bert_search(q, k):
    q_emb = bert_model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, corpus_embeddings)[0]
    idx = torch.topk(scores, k).indices.tolist()
    return [code_list[i] for i in idx]

def precision_at_k(model_fn, k):
    hits = 0
    total = len(gt_df)

    for _, row in tqdm(gt_df.iterrows(), total=total):
        q = row["diagnosis_text"]
        true_code = str(row["icd_code"]).strip()

        try:
            preds = model_fn(q, k)
            # ★ 正確比對方式：用 startswith
            if any(p.startswith(true_code) for p in preds):
                hits += 1
        except:
            pass

    return hits / total

ks = [1, 5, 10]
for k in ks:
    p = precision_at_k(bert_search, k)
    print(f"BERT Precision@{k} = {p:.4f}")
