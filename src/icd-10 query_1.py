# ============================================
# 中文查詢程式：ICD-10-CM Ontology 查詢工具
# ============================================

from rdflib import Graph, Namespace, RDFS, SKOS



# ===== 1️⃣ 載入 OWL 檔 =====
owl_file = r"C:\Users\User\Desktop\全球人壽產學\ICD10\icd_with_labels.rdf"

g = Graph()
g.parse(owl_file, format="xml")

ICD = Namespace("http://www.semanticweb.org/user/ontologies/2025/10/untitled-ontology-20")   # ⚠️ 要和你建 ontology 時設定的 IRI 一致


# ===== 2️⃣ 工具函式 =====
def get_parent(code_uri):
    """查詢節點的直接父節點 URI"""
    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?parent
    WHERE {{
        <{code_uri}> rdfs:subClassOf ?parent .
    }}
    """
    for row in g.query(q):
        return str(row.parent)
    return None


def get_full_path(code_uri):
    """遞迴往上追 rdfs:subClassOf 直到根節點，組合完整階層路徑"""
    path = []
    current = code_uri
    while current:
        code_id = current.split("/")[-1]
        path.insert(0, code_id)
        current = get_parent(current)
    return " → ".join(path)


def _local_id_from_uri(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1]

def search_icd_label(query: str):
    """
    中英皆可；兼容 rdfs:label 與 skos:prefLabel。
    會回傳：code_id, shown_label(最匹配的), zh_label, en_label, full_path
    """
    safe = query.replace('"', '\\"')
    results = []

    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?code ?hitLabel ?zh ?en
    WHERE {{
      # 命中用的 label：rdfs 或 skos 都算（不限制語言，方便現在就能用）
      {{
        ?code rdfs:label ?hitLabel .
      }} UNION {{
        ?code skos:prefLabel ?hitLabel .
      }}
      FILTER(CONTAINS(LCASE(STR(?hitLabel)), LCASE("{safe}")))

      # 將來若你加了語言標籤，這兩段會盡量帶回對應中英名稱
      OPTIONAL {{
        {{
          ?code rdfs:label ?zh .
        }} UNION {{
          ?code skos:prefLabel ?zh .
        }}
        FILTER(LANG(?zh) = "zh-TW" || LANG(?zh) = "zh" || LANG(?zh) = "")
      }}

      OPTIONAL {{
        {{
          ?code rdfs:label ?en .
        }} UNION {{
          ?code skos:prefLabel ?en .
        }}
        FILTER(LANG(?en) = "en")
      }}
    }}
    """

    for row in g.query(q):
        code_uri  = str(row.code)
        code_id   = _local_id_from_uri(code_uri)
        hit_label = str(row.hitLabel)
        zh_label  = str(row.zh) if row.zh else None
        en_label  = str(row.en) if row.en else None
        path      = get_full_path(code_uri)

        # 顯示用：若有中文優先中文；否則用命中的 label
        shown = zh_label or hit_label
        results.append((code_id, shown, zh_label, en_label, path))

    return results


# ===== 3️⃣ 主程式：互動查詢 =====
if __name__ == "__main__":
    print(" ICD 名稱查詢系統（Ontology 版本）")
    print("輸入中文或英文病名（例如：霍亂 / cholera / 腸胃炎 / diabetes），輸入 q 離開。\n")

    while True:
        query = input("請輸入病名：").strip()
        if query.lower() == "q":
            break

        matches = search_icd_label(query)  # << 重點：用「兼容 rdfs/skos、兼容中英」的版本
        if not matches:
            print(" 查無結果，請換個詞或檢查資料。\n")
            continue

        print(f"\n 查詢結果（關鍵字：{query}）")
        print("--------------------------------------------------")
        for code_id, shown, zh_label, en_label, path in matches:
            print(f"ICD代碼: {code_id}")
            # shown 是優先顯示用的名稱（若有中文就中文，否則命中標籤）
            print(f"顯示名稱: {shown}")
            if zh_label:
                print(f"中文名稱: {zh_label}")
            if en_label:
                print(f"英文名稱: {en_label}")
            print(f"階層路徑: {path}\n")
        print("--------------------------------------------------\n")
