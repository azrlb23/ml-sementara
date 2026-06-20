# === Cell 1 ===import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. BACA DATA (Gunakan Pemenang Clustering: Standard K-Means)
df = pd.read_csv('../data/Labeled/hasildata_kmeans-qlde.csv')

# Pisahkan Fitur dan Target
fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

# Splitting Data (80% Train, 20% Test) - Menggunakan 100% baris data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. SCALING DATA (Langkah wajib untuk ANN)
scaler_ann = StandardScaler()
X_train_scaled = scaler_ann.fit_transform(X_train)
X_test_scaled = scaler_ann.transform(X_test)

# 3. TRAINING MODEL ANN
# Menggunakan max_iter=500 agar model punya cukup waktu belajar dan konvergen
model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model_ann.fit(X_train_scaled, y_train)

# Evaluasi
prediksi_ann = model_ann.predict(X_test_scaled)
print("=== CLASSIFICATION REPORT: ANN (QLDE) ===\n")
print(classification_report(y_test, prediksi_ann))

# 4. EXPORT SCALER & MODEL KE FOLDER 'models'
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler_ann, '../models/scaler_ann_qlde.pkl')
joblib.dump(model_ann, '../models/model_ann_classification_qlde.pkl')
print("\n[SUCCESS] Scaler & Model ANN diekspor ke folder '../models/'!")

