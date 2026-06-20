# === Cell 1 ===import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Baca Data
filepath = '../data/Labeled/hasildata_kmeans-qlde.csv'
df = pd.read_csv(filepath)

fitur = [f'Var{i}' for i in range(1, 12)]
X = df[fitur]
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Batasi Sampel SVM maksimal 2000
X_train_svm = X_train[:2000]
y_train_svm = y_train[:2000]

# 3. Scaling Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_svm)
X_test_scaled = scaler.transform(X_test)

# 4. Latih SVM (Gunakan class_weight='balanced' untuk mengatasi kelompok langka)
model_svm_qlde = SVC(kernel='rbf', random_state=42, class_weight='balanced')
model_svm_qlde.fit(X_train_scaled, y_train_svm)

# 5. Evaluasi
prediksi_svm_qlde = model_svm_qlde.predict(X_test_scaled)
print("=== CLASSIFICATION REPORT: SVM (QLDE) ===\n")
print(classification_report(y_test, prediksi_svm_qlde, zero_division=0))

# 5. EXPORT SCALER & MODEL KE FOLDER 'models'
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler_svm_qlde.pkl')
joblib.dump(model_svm_qlde, '../models/model_svm_classification_qlde.pkl')
print("\n[SUCCESS] Scaler & Model SVM diekspor ke folder '../models/'!")

