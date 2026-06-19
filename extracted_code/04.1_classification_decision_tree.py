# === Cell 1 ===import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

# 1. BACA DATA (Gunakan Pemenang Clustering: Standard K-Means)
df = pd.read_csv('../data/Labeled/hasildata_kmeans-standard.csv')

# Pisahkan Fitur dan Target
fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

# Splitting Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TRAINING MODEL DECISION TREE
# Batasi kedalaman maksimal agar aturan bisnis tidak terlalu rumit dibaca
model_dt = DecisionTreeClassifier(random_state=42, max_depth=4)
model_dt.fit(X_train, y_train)

# Evaluasi
prediksi_dt = model_dt.predict(X_test)
print("=== CLASSIFICATION REPORT: DECISION TREE (STANDARD K-MEANS) ===\n")
print(classification_report(y_test, prediksi_dt))

# 3. EKSTRAKSI ATURAN BISNIS
aturan_bisnis = export_text(model_dt, feature_names=fitur)
print("=== BUSINESS RULES (ATURAN LOGIKA PELANGGAN) ===\n")
print(aturan_bisnis)

# 4. EXPORT MODEL KE FOLDER 'models'
os.makedirs('../models', exist_ok=True)
joblib.dump(model_dt, '../models/model_dt_classification.pkl')
print("\n[SUCCESS] Model Decision Tree diekspor ke '../models/model_dt_classification.pkl'")

