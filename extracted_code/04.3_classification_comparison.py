# === Cell 1 ===import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

# 1. Baca Data Hasil QLDE
filepath = '../data/Labeled/hasildata_kmeans-qlde.csv'
df = pd.read_csv(filepath)

fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Latih Model Decision Tree
model_dt_qlde = DecisionTreeClassifier(random_state=42, max_depth=4)
model_dt_qlde.fit(X_train, y_train)

# 3. Evaluasi Model
prediksi_dt = model_dt_qlde.predict(X_test)
print("=== CLASSIFICATION REPORT: DECISION TREE (QLDE) ===\n")
print(classification_report(y_test, prediksi_dt, zero_division=0))

# 4. Cetak Aturan Bisnis
aturan_bisnis = export_text(model_dt_qlde, feature_names=fitur)
print("\n=== BUSINESS RULES QLDE ===\n")
print(aturan_bisnis)

