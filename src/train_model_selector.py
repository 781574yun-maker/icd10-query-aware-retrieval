# train_model_selector.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =====================================
# 1. 讀取資料
# =====================================
df = pd.read_csv("train_selector_data.csv", encoding="utf-8")

# 你要使用的 features（除掉 query 與 row_id）
feature_cols = [
    c for c in df.columns
    if c not in ["row_id", "query", "true_best_model"]
]

X = df[feature_cols]
y = df["true_best_model"]

# =====================================
# 2. 標籤編碼（string → int）
# =====================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =====================================
# 3. 切 train / test
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================
# 4. 訓練 RandomForest（最穩定）
# =====================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =====================================
# 5. 評估
# =====================================
y_pred = model.predict(X_test)

print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# =====================================
# 6. Feature importance
# =====================================
importances = model.feature_importances_
ImportanceDF = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance ===")
print(ImportanceDF)

# =====================================
# 7. 儲存模型（selector）
# =====================================
joblib.dump(model,r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\model_selector.pkl")
joblib.dump(le, r"C:\Users\User\Desktop\全球人壽產學\icd_project\models\label_encoder.pkl")

print("\n模型已儲存：model_selector.pkl")
print("Label Encoder 已儲存：label_encoder.pkl")
