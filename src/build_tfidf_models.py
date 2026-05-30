# ============================================================
# build_tfidf_models.py
# 產生兩套 TF-IDF 模型：
#   1. jieba 分詞
#   2. spaCy 分詞
# ============================================================

import pandas as pd
import jieba
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------
# 1. 讀取 ICD 詞庫
# ------------------------------------------------------------

print("Loading ICD vocabulary...")
df = pd.read_csv("icd_vocab_new.csv").fillna("")

# text_for_search = 中文 + 英文
df["text_for_search"] = (df["zh_label"] + " " + df["en_label"]).astype(str).str.strip()
corpus = df["text_for_search"].tolist()


# ------------------------------------------------------------
# 2. 準備 spaCy（中文模型）
# ------------------------------------------------------------

print("Loading spaCy zh_core_web_sm ... (may take a few seconds)")
nlp = spacy.load("zh_core_web_sm")


def spacy_cut(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc])


# ------------------------------------------------------------
# 3. jieba 分詞 TF-IDF
# ------------------------------------------------------------

print("Building jieba TF-IDF model...")

jieba_cut_corpus = [" ".join(jieba.cut(t)) for t in corpus]

vectorizer_jieba = TfidfVectorizer()
tfidf_matrix_jieba = vectorizer_jieba.fit_transform(jieba_cut_corpus)

# ------------------------------------------------------------
# 4. spaCy 分詞 TF-IDF
# ------------------------------------------------------------

print("Building spaCy TF-IDF model...")

spacy_cut_corpus = [spacy_cut(t) for t in corpus]

vectorizer_spacy = TfidfVectorizer()
tfidf_matrix_spacy = vectorizer_spacy.fit_transform(spacy_cut_corpus)


# ------------------------------------------------------------
# 5. 存檔（pickle）
# ------------------------------------------------------------

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f" Saved: {filename}")


print("\nSaving models...")

save_pickle(vectorizer_jieba, "tfidf_jieba_vectorizer.pkl")
save_pickle(tfidf_matrix_jieba, "tfidf_jieba_matrix.pkl")

save_pickle(vectorizer_spacy, "tfidf_spacy_vectorizer.pkl")
save_pickle(tfidf_matrix_spacy, "tfidf_spacy_matrix.pkl")

print("\nAll TF-IDF models built successfully!")
