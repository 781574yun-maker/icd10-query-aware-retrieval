# oracle_precision.py
import pandas as pd

CSV_PATH = r"C:\Users\User\Desktop\全球人壽產學\icd_project\model_per_query.csv"
df = pd.read_csv(CSV_PATH)

def oracle_precision_at_k(df, k):
    total = len(df)
    hits = 0

    bert_col   = f"bert_hit{k}"
    jieba_col  = f"hybrid_jieba_hit{k}"
    spacy_col  = f"hybrid_spacy_hit{k}"

    for _, row in df.iterrows():
        # 只要三種模型中，有任何一個 hit，就算 oracle 命中
        if (row[bert_col] == 1) or (row[jieba_col] == 1) or (row[spacy_col] == 1):
            hits += 1

    return hits / total


if __name__ == "__main__":
    print("=== Oracle Precision（最理論上限）===\n")
    for k in [1, 5, 10]:
        p = oracle_precision_at_k(df, k)
        print(f"Oracle Precision@{k} = {p:.4f}")
