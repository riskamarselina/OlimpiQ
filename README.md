# 🏆 OlympiQ — Sistem Seleksi Tahap Awal Olimpiade Matematika

> Sistem klasifikasi berbasis **Machine Learning (Random Forest)** untuk membantu guru mengidentifikasi kesiapan siswa dalam mengikuti olimpiade matematika.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange.svg)](https://scikit-learn.org)
[![Akurasi](https://img.shields.io/badge/Akurasi-95.53%25-brightgreen.svg)](#)

---

## 📋 Deskripsi Proyek

Sistem ini membantu guru dan pemangku kepentingan pendidikan untuk **mengidentifikasi siswa yang berpotensi mengikuti olimpiade matematika** secara otomatis dan berbasis data. Sistem mengklasifikasikan siswa ke dalam 3 kategori:

| Status | Keterangan |
|--------|------------|
| 🏆 **Siap Olimpiade** | Kompetensi tinggi, siap mengikuti seleksi |
| ⚡ **Potensial** | Ada potensi, perlu pembinaan intensif |
| ❌ **Tidak Siap** | Perlu penguatan kompetensi dasar terlebih dahulu |

---

## 🧠 Fitur Input Sistem

| Kolom | Keterangan | Bobot |
|-------|------------|-------|
| `NUM_ALJ` | Skor Numerasi Aljabar (0–100) | 20% |
| `NUM_GEO` | Skor Numerasi Geometri (0–100) | 15% |
| `NUM_BIL` | Skor Numerasi Bilangan (0–100) | 20% |
| `NUM_DAT` | Skor Data & Ketidakpastian (0–100) | 15% |
| `NUM_L3`  | Skor Menalar (0–100) | 15% |
| `LIT`     | Skor Literasi (0–100) | 15% |

---

## 📊 Performa Model

| Metrik | Nilai |
|--------|-------|
| Algoritma | Random Forest Classifier |
| Akurasi Test Set | **95.53%** |
| Cross-Validation (5-fold) | **95.28% ± 0.36%** |
| Data Training | 23.647 siswa |
| Data Testing | 5.912 siswa |

---

## 🗂️ Struktur Proyek

```
olimpiade-system/
├── 📂 app/
│   └── app.py                    ← Aplikasi Streamlit utama
├── 📂 data/
│   └── dataset.xlsx              ← Dataset (30.000 data siswa)
├── 📂 model/                     ← Folder model (diisi setelah training)
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── thresholds.pkl
│   └── model_meta.pkl
├── 📂 notebooks/
│   └── OlympiQ_Training.ipynb   ← Notebook Google Colab
├── train_model.py                ← Script training model lokal
├── requirements.txt              ← Daftar library Python
└── README.md                     ← Dokumentasi ini
```

---

## 🚀 Langkah-Langkah Penggunaan

### 🔬 OPSI A: Training di Google Colab (Direkomendasikan)

**Langkah 1: Buka Google Colab**
1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Klik **File → Upload notebook**
3. Upload file `notebooks/OlympiQ_Training.ipynb`

**Langkah 2: Upload Dataset**
- Jalankan sel bagian "Upload & Load Dataset"
- Klik tombol upload dan pilih file `dataset.xlsx`

**Langkah 3: Jalankan Semua Sel**
- Klik **Runtime → Run all** (Ctrl+F9)
- Tunggu hingga semua sel selesai dieksekusi

**Langkah 4: Download Model**
- Jalankan sel terakhir di bagian "Simpan Model"
- File `model_olimpiade.zip` akan terdownload otomatis
- **Extract ZIP** dan letakkan isi folder `model/` ke dalam proyek kamu

---

### 💻 OPSI B: Training Lokal

**Langkah 1: Clone Repositori**
```bash
git clone https://github.com/USERNAME/olimpiade-system.git
cd olimpiade-system
```

**Langkah 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Langkah 3: Jalankan Training**
```bash
python train_model.py
```

**Langkah 4: Jalankan Aplikasi Streamlit**
```bash
streamlit run app/app.py
```

Buka browser di `http://localhost:8501`

---

### ☁️ OPSI C: Deploy ke Streamlit Cloud

**Langkah 1: Push ke GitHub**
```bash
git init
git add .
git commit -m "Initial commit: OlympiQ sistem seleksi olimpiade"
git branch -M main
git remote add origin https://github.com/USERNAME/olimpiade-system.git
git push -u origin main
```

> ⚠️ **Penting:** Pastikan folder `model/` (berisi file `.pkl`) sudah ter-push ke GitHub sebelum deploy!

**Langkah 2: Deploy di Streamlit Cloud**
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **"New app"**
4. Isi formulir:
   - **Repository:** `USERNAME/olimpiade-system`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
5. Klik **"Deploy!"**
6. Tunggu beberapa menit hingga aplikasi online

---

## 📄 Format File Excel untuk Upload Batch

Saat menggunakan fitur **Upload File Excel** di aplikasi, pastikan file Excel kamu memiliki format kolom berikut:

```
| Nama       | NUM_ALJ | NUM_GEO | NUM_BIL | NUM_DAT | NUM_L3 | LIT   |
|------------|---------|---------|---------|---------|--------|-------|
| Andi Budi  | 88.5    | 91.0    | 85.0    | 87.5    | 90.0   | 92.0  |
| Rina Cahya | 65.0    | 70.0    | 62.5    | 68.0    | 66.0   | 71.5  |
```

> Kolom `Nama` bersifat **opsional**. Kolom skor wajib ada persis dengan nama tersebut.

---

## 🔧 Penjelasan Teknis

### Algoritma: Random Forest Classifier

Random Forest dipilih karena:
- **Robust terhadap outlier** (data skor yang ekstrem)
- **Tidak mudah overfitting** berkat mekanisme bagging
- **Memberikan feature importance** sehingga transparan
- **Akurasi tinggi** untuk data tabular dengan skala numerik

### Pelabelan Data

Label dibuat menggunakan **metode percentile berbasis distribusi data nyata**:
- `Tidak Siap` → Skor total < Persentil ke-33
- `Potensial` → Persentil ke-33 ≤ Skor total < Persentil ke-66
- `Siap Olimpiade` → Skor total ≥ Persentil ke-66

### Preprocessing

- Missing values di-drop (238 baris, <1% dari total data)
- Normalisasi menggunakan **StandardScaler**
- Split data: 80% latih / 20% uji dengan stratified sampling

---

## 👨‍💻 Kontributor

- [Nama Anggota 1] — Data Preprocessing & Modeling
- [Nama Anggota 2] — UI/UX Streamlit
- [Nama Anggota 3] — Evaluasi & Dokumentasi

**Mata Kuliah:** Data Mining  
**Program Studi:** Pendidikan Matematika  
**Tahun:** 2024/2025

---

## 📜 Lisensi

MIT License — Bebas digunakan untuk keperluan pendidikan.
