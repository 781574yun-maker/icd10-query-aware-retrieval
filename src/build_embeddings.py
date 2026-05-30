# build_embeddings.py
# 將 ICD 文字轉換為 BERT 向量，並儲存為 icd_embeddings.pkl
# 執行方式: python build_embeddings.py

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ===== 路徑設定 =====
CSV_FILE = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
EMBED_FILE = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_embeddings.pkl"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"

print("[INFO] 載入 ICD 詞庫中...")
df = pd.read_csv(CSV_FILE, encoding="utf-8").fillna("")
df["text_for_search"] = (df["zh_label"].astype(str) + " " + df["en_label"].astype(str)).str.strip()

# ===== 載入模型 =====
print(f"[INFO] 載入 BERT 模型：{MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

# ===== 產生向量 =====
corpus = df["text_for_search"].tolist()
print(f"[INFO] 正在產生 embeddings，共 {len(corpus)} 筆資料...")

embeddings = []
for text in tqdm(corpus):
    emb = model.encode(text)
    embeddings.append(emb)

# ===== 儲存向量到檔案 =====
print(f"[INFO] 儲存向量檔案至: {EMBED_FILE}")
with open(EMBED_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print("向量建立完成！已存入 icd_embeddings.pkl")
