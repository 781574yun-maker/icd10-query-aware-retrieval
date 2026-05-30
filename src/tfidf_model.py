import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 詞庫路徑：請根據你的檔案位置調整
csv_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab_new.csv"

# 讀取詞庫（避免 NaN）
df = pd.read_csv(csv_file, encoding="utf-8")
df = df.fillna("")

# 合併中英文欄位做 TF-IDF 文本
texts = df["zh_label"] + " " + df["en_label"]

# jieba 斷詞
texts_cut = [" ".join(jieba.cut(text)) for text in texts]

# 建立 TF-IDF 模型
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts_cut)

# 互動查詢模式
print("輸入 q 離開\n")

while True:
    query = input("\n請輸入中文查詢詞：")

    #  退出條件
    if query.lower() == "q":
        print("bye!")
        break
    
    # 防呆：空白輸入
    if not query.strip():
        print("請輸入有效文字")
        continue

    # 斷詞 & vector
    query_cut = " ".join(jieba.cut(query))
    query_vec = vectorizer.transform([query_cut])

    # cosine similarity
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k = similarities.argsort()[::-1][:5]

    # 顯示結果
    print("\n查詢結果（Top-5）：\n")
    for i in top_k:
        print(f"{df.loc[i,'code_id']} | {df.loc[i,'zh_label']} ({df.loc[i,'en_label']}) "
              f"| 相似度：{similarities[i]:.4f}")
