# ============================================
# 檔案名稱：build_vocab_from_owl.py
# 功能：從 icd_with_labels.rdf 擷取所有 ICD 節點 → 儲存為 icd_vocab.csv
# ============================================

from rdflib import Graph, Namespace, RDF, RDFS, SKOS, OWL, URIRef
import pandas as pd

# 絕對路徑設定：請確認這條路徑是你電腦上的 RDF 檔案實際位置
rdf_path = r"C:\Users\User\Desktop\全球人壽產學\ICD10\icd_with_labels.rdf"

# 讀取 RDF 檔
g = Graph()
g.parse(rdf_path, format="xml")

# 設定 Namespace
ICD = Namespace("http://www.semanticweb.org/user/ontologies/2025/10/untitled-ontology-20")

# 取得父節點 URI（不使用 SPARQL）
def get_parent(code_uri):
    for _, _, parent in g.triples((URIRef(code_uri), RDFS.subClassOf, None)):
        return str(parent)
    return None

# 取得完整階層路徑
def get_full_path(code_uri):
    path = []
    current = code_uri
    while current:
        code_id = current.split("/")[-1]
        path.insert(0, code_id)
        current = get_parent(current)
    return " → ".join(path)

# 擷取中英文 label
def extract_label(code_uri, lang):
    labels = list(g.objects(URIRef(code_uri), RDFS.label)) + list(g.objects(URIRef(code_uri), SKOS.prefLabel))
    for label in labels:
        if lang == "zh":
            if not label.language or label.language in ["zh", "zh-TW"]:
                return str(label)
        elif lang == "en":
            if label.language == "en":
                return str(label)
    return None

# 建立詞庫表格
vocab_data = []

for s in g.subjects(RDF.type, OWL.Class):
    code_uri = str(s)
    code_id = code_uri.split("#")[-1]
    zh_label = extract_label(code_uri, "zh")
    en_label = extract_label(code_uri, "en")
    full_path = get_full_path(code_uri)
    text_for_search = " ".join(filter(None, [zh_label, en_label]))
    vocab_data.append({
        "code_id": code_id,
        "zh_label": zh_label,
        "en_label": en_label,
        "full_path": full_path,
        "text_for_search": text_for_search
    })

df_vocab = pd.DataFrame(vocab_data)

# 輸出為 UTF-8-sig 編碼的 CSV（Excel 開不亂碼）
csv_output = r"C:\Users\User\Desktop\全球人壽產學\icd_project\icd_vocab.csv"
df_vocab.to_csv(csv_output, index=False, encoding="utf-8-sig")

print("詞庫建立完成，已輸出至：", csv_output)
