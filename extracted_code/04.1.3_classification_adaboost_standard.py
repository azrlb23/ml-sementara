# === Cell 1 ===import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

# 1. BACA DATA (Gunakan Pemenang Clustering: Standard K-Means)
df = pd.read_csv('../data/Labeled/hasildata_kmeans-standard.csv')

# Pisahkan Fitur dan Target
fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

# Splitting Data (80% Train, 20% Test) - Menggunakan 100% baris data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TRAINING MODEL ADABOOST
model_adaboost = AdaBoostClassifier(random_state=42)
model_adaboost.fit(X_train, y_train)

# Evaluasi
prediksi_adaboost = model_adaboost.predict(X_test)
print("=== CLASSIFICATION REPORT: ADABOOST (STANDARD K-MEANS) ===\n")
print(classification_report(y_test, prediksi_adaboost))

# 3. EXPORT MODEL KE FOLDER 'models'
os.makedirs('../models', exist_ok=True)
joblib.dump(model_adaboost, '../models/model_adaboost_classification_standard.pkl')
print("\n[SUCCESS] Model AdaBoost diekspor ke '../models/model_adaboost_classification_standard.pkl'")

