"""
=============================================================
  SISTEM SELEKSI OLIMPIADE MATEMATIKA
  Script: train_model.py
  Deskripsi: Melatih model Random Forest Classifier untuk
             klasifikasi kesiapan olimpiade siswa
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi ────────────────────────────────────────────
DATA_PATH = 'dataset.xlsx'
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = ['NUM_ALJ', 'NUM_GEO', 'NUM_BIL', 'NUM_DAT', 'NUM_L3', 'LIT']
FEATURE_NAMES = {
    'NUM_ALJ': 'Numerasi Aljabar',
    'NUM_GEO': 'Numerasi Geometri',
    'NUM_BIL': 'Numerasi Bilangan',
    'NUM_DAT': 'Data & Ketidakpastian',
    'NUM_L3':  'Skor Menalar',
    'LIT':     'Literasi'
}

WEIGHTS = {
    'NUM_ALJ': 0.20,
    'NUM_GEO': 0.15,
    'NUM_BIL': 0.20,
    'NUM_DAT': 0.15,
    'NUM_L3':  0.15,
    'LIT':     0.15
}

CLASS_NAMES = {0: 'Tidak Siap', 1: 'Potensial', 2: 'Siap Olimpiade'}

print("=" * 60)
print("  PELATIHAN MODEL SELEKSI OLIMPIADE MATEMATIKA")
print("=" * 60)

# ─── 1. Load & Preprocessing ────────────────────────────────
print("\n[1/5] Memuat dataset...")
df = pd.read_excel(DATA_PATH)
df.columns = FEATURE_COLS
df = df.dropna()
print(f"    Total data valid: {len(df):,} baris")

# ─── 2. Buat Label ──────────────────────────────────────────
print("\n[2/5] Membuat label klasifikasi...")
df['TOTAL_SCORE'] = sum(df[col] * weight for col, weight in WEIGHTS.items())

p33 = df['TOTAL_SCORE'].quantile(0.33)
p66 = df['TOTAL_SCORE'].quantile(0.66)

print(f"    Threshold Tidak Siap   : < {p33:.2f}")
print(f"    Threshold Potensial    : {p33:.2f} - {p66:.2f}")
print(f"    Threshold Siap Olimpiade: >= {p66:.2f}")

df['LABEL'] = df['TOTAL_SCORE'].apply(
    lambda s: 2 if s >= p66 else (1 if s >= p33 else 0)
)

dist = df['LABEL'].value_counts().sort_index()
for label, count in dist.items():
    print(f"    {CLASS_NAMES[label]}: {count:,} ({count/len(df)*100:.1f}%)")

# ─── 3. Split Data ──────────────────────────────────────────
print("\n[3/5] Membagi data latih dan uji...")
X = df[FEATURE_COLS]
y = df['LABEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Data latih: {len(X_train):,} | Data uji: {len(X_test):,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 4. Training & Evaluasi ─────────────────────────────────
print("\n[4/5] Melatih Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sc, y_train)

y_pred = rf.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)

print(f"\n    Akurasi Test Set : {acc:.4f} ({acc*100:.2f}%)")

cv_scores = cross_val_score(
    rf, X_train_sc, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
print(f"    Cross-Val (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n    Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Tidak Siap', 'Potensial', 'Siap Olimpiade']
))

print("    Feature Importance:")
for feat, imp in sorted(
    zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1]
):
    bar = "█" * int(imp * 40)
    print(f"    {FEATURE_NAMES[feat]:<25} {imp:.4f}  {bar}")

# ─── 5. Simpan Model ────────────────────────────────────────
print("\n[5/5] Menyimpan model...")
joblib.dump(rf,      f"{MODEL_DIR}/rf_model.pkl")
joblib.dump(scaler,  f"{MODEL_DIR}/scaler.pkl")
joblib.dump({'p33': p33, 'p66': p66}, f"{MODEL_DIR}/thresholds.pkl")

meta = {
    'accuracy':     float(acc),
    'cv_mean':      float(cv_scores.mean()),
    'cv_std':       float(cv_scores.std()),
    'n_train':      int(len(X_train)),
    'n_test':       int(len(X_test)),
    'feature_cols': FEATURE_COLS,
    'feature_names': FEATURE_NAMES,
    'weights':      WEIGHTS,
    'thresholds':   {'p33': float(p33), 'p66': float(p66)},
    'feature_importance': dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
}
joblib.dump(meta, f"{MODEL_DIR}/model_meta.pkl")

print(f"    Model disimpan di folder: {MODEL_DIR}/")
print("\n" + "=" * 60)
print("  PELATIHAN SELESAI! Jalankan: streamlit run app/app.py")
print("=" * 60)
