# bert_model.py
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# 1) 讀取詞庫 CSV
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"
df = pd.read_csv(csv_file, encoding="utf-8").fillna("")
# 你的欄位名稱若不同，這裡要對應調整
# 例如：code_id, zh_label, en_label
df["text_for_search"] = (df["zh_label"].astype(str) + " " + df["en_label"].astype(str)).str.strip()

# 2) 載入模型（多語言）
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

# 3) 預先編碼整個語料
corpus = df["text_for_search"].tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)


# 4) 封裝成查詢函式
def bert_search(query: str, top_k: int = 5):
    """回傳 [(code_id, '中文（英文）', score_float), ...]"""
    query = query.strip()
    if not query:
        return []
    q_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, corpus_embeddings)[0]
    k = min(top_k, len(df))
    scores, indices = cos_scores.topk(k=k)

    results = []
    for s, idx in zip(scores, indices):
        row = df.iloc[int(idx)]
        code = row.get("code_id", "")
        zh = row.get("zh_label", "")
        en = row.get("en_label", "")
        results.append((code, f"{zh} ({en})", float(s.item())))
    return results

# 5) 互動式查詢直到輸入 q 結束
if __name__ == "__main__":
    print("輸入 q 退出查詢")

    while True:
        user_input = input("\n請輸入中文查詢詞：").strip()
        if user_input.lower() == "q":
            print("bye！")
            break
        if not user_input:
            print("請輸入有效的詞")
            continue

        results = bert_search(user_input, top_k=5)
        if not results:
            print("（沒有結果）")
            continue

        print("\n查詢結果（Top-5）：\n")
        for code, name, score in results:
            print(f"{code} | {name} | 相似度：{score:.4f}")

