import pandas as pd

# ===== 1. 設定路徑 =====
input_file = r"C:\Users\User\Desktop\全球人壽產學\ICD10\理賠建檔_ICD10_20250903.xlsx"
output_file = r"C:\Users\User\Desktop\全球人壽產學\icd_project\clean_claim_gt.csv"

# ===== 2. 讀取 Excel =====
df = pd.read_excel(input_file).fillna("")

records = []

current_diag_text = None  # 暫存診斷內容（因為它在上一列）

# ===== 3. 逐列掃描 =====
for idx, row in df.iterrows():

    col_name = str(row["欄位名稱"]).strip()
    val1 = str(row["建檔值1"]).strip()
    val2 = str(row["建檔值2"]).strip()

    # ① 遇到「診斷內容」→ 儲存為 current_diag_text
    if col_name == "診斷內容":
        if val1 != "" and val1 != "NULL":
            current_diag_text = val1
        continue

    # ② 遇到「診斷代碼/名稱0 / 1 / 2 …」→ 是 ICD
    if col_name.startswith("診斷代碼/名稱"):

        if current_diag_text is None:
            continue  # 理論上不會發生

        icd_code = val1 if val1 not in ["", "NULL"] else None
        icd_name = val2 if val2 not in ["", "NULL"] else ""

        if icd_code:
            records.append({
                "diagnosis_text": current_diag_text,
                "icd_code": icd_code,
                "icd_name": icd_name
            })

# ===== 4. 轉成 DataFrame 並輸出 =====
out_df = pd.DataFrame(records)
out_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("清理完成！輸出：", output_file)
print("總筆數：", len(out_df))
