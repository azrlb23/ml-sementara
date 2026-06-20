# === Cell 1 ===import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. BACA DATA (Gunakan Pemenang Clustering: Standard K-Means)
df = pd.read_csv('../data/Labeled/hasildata_kmeans-standard.csv')

fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membatasi sampel data latih maksimal 2000 baris untuk mencegah processing delays
X_train_svm = X_train[:2000]
y_train_svm = y_train[:2000]

# 2. SCALING DATA (Langkah wajib untuk algoritma SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_svm)
X_test_scaled = scaler.transform(X_test)

# 3. TRAINING MODEL SVM
model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train_scaled, y_train_svm)

# Evaluasi
prediksi_svm = model_svm.predict(X_test_scaled)
print("=== CLASSIFICATION REPORT: SVM (STANDARD K-MEANS) ===\n")
print(classification_report(y_test, prediksi_svm))

# 4. EXPORT SCALER & MODEL KE FOLDER 'models'
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler_svm_standard.pkl')
joblib.dump(model_svm, '../models/model_svm_classification_standard.pkl')
print("\n[SUCCESS] Scaler & Model SVM diekspor ke folder '../models/'!")

