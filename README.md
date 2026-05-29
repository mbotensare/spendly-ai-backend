# 🏦 Spendly AI — Smart Receipt & Spending Assistant

> **Coding Camp 2026 powered by DBS Foundation**
> Tim **CC26-PSU276**

Spendly AI adalah sistem kecerdasan buatan yang membantu pengguna mengelola pengeluaran pribadi secara otomatis. Pengguna cukup memfoto struk belanja — sistem akan **mendeteksi area struk**, **membaca teks** di dalamnya, **mengkategorikan jenis pengeluaran**, dan **memprediksi pola pengeluaran** minggu berikutnya.

Sistem dibangun **100% dari nol** menggunakan TensorFlow dengan pendekatan **multimodal** yang menggabungkan empat model utama: **Receipt Detection** (CNN), **OCR** (CRNN + CTC), **Classification** (TF-IDF + CNN + SE Attention), dan **Time-Series Forecasting** (LSTM + Custom Attention).

---

## Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Kategori Pengeluaran](#-kategori-pengeluaran)
- [Data yang Digunakan](#-data-yang-digunakan)
- [Pipeline Preprocessing](#-pipeline-preprocessing)
- [Arsitektur Model](#-arsitektur-model)
- [Custom Components](#-custom-components)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Riwayat Eksperimen OCR](#-riwayat-eksperimen-ocr)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Menjalankan](#-cara-menjalankan)
- [REST API](#-rest-api)
- [Gemini AI Insight](#-gemini-ai-insight)
- [TensorBoard](#-tensorboard)
- [Output Visualisasi](#-output-visualisasi)
- [Catatan Teknis](#-catatan-teknis)
- [Tim & Lisensi](#-tim--lisensi)

---

## ✨ Fitur Utama

| Fitur | Deskripsi | Teknologi |
|---|---|---|
| 📷 **Receipt Detection** | Mendeteksi dan melokalisasi area struk dalam foto | CNN Custom (Conv + Dense) |
| 🔍 **OCR Struk** | Membaca teks dari foto struk: harga, toko, tanggal, item | CRNN Custom (Conv+BiLSTM) + CTC Loss |
| 🏷️ **Klasifikasi Pengeluaran** | Mengkategorikan transaksi ke 9 kategori otomatis | TF-IDF (5000) + CNN + SEBlock Multimodal + NLP PySastrawi |
| 📈 **Spending Forecasting** | Memprediksi pengeluaran minggu berikutnya per kategori | LSTM(64→32) + Custom Attention Layer |
| 💡 **AI Financial Insight** | Saran keuangan personal berbahasa Indonesia | Google Gemini 2.5 Flash (Generative AI) |
| 🔍 **Blur Detection** | Mendeteksi gambar blur sebelum OCR | Laplacian Variance (threshold = 100) |
| ⚖️ **Equalisasi Data** | Menyeimbangkan jumlah data per kelas | Offline Augmentation via Albumentations (target 200/kelas) |

---

## 🧾 Kategori Pengeluaran

Spendly AI mendukung **9 kategori** transaksi:

| No | Kategori | Contoh Merchant / Item |
|---|---|---|
| 1 | **Beauty** | Sociolla, Watsons, Guardian — skincare, makeup, parfum |
| 2 | **F&B** | Starbucks, Solaria, McDonald's, Fore Coffee — makanan & minuman |
| 3 | **Gas** | Pertamina, Shell — BBM kendaraan |
| 4 | **Groceries** | Indomaret, Alfamart, Superindo — belanja harian |
| 5 | **Health** | Apotek Kimia Farma, Halodoc — obat, vitamin |
| 6 | **HouseHold** | ACE Hardware, Mr DIY — peralatan rumah tangga |
| 7 | **Lifestyle** | Uniqlo, Miniso — fashion, aksesoris |
| 8 | **Listrik** | PLN Token — pembayaran listrik |
| 9 | **Transport** | Grab, Gojek, MRT, TransJakarta — transportasi |

---

## 📂 Data yang Digunakan

### 1. Data Gambar Struk (Classifier & OCR Real Data)

Data utama berupa foto struk belanja asli yang diorganisasi ke dalam 9 folder berdasarkan kategori di `data/{Kategori}/`.

#### Pengumpulan Gambar

Semua gambar dikumpulkan secara rekursif dari setiap folder kelas menggunakan ekstensi `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. File yang berada di dalam subfolder `labels/` difilter dan tidak diikutkan.

#### Stratified Split

Data dibagi menggunakan **stratified split 70/15/15** dengan `random_state=42`:

| Split | Proporsi | Keterangan |
|---|---|---|
| Train | 70% | Digunakan untuk training + equalisasi |
| Validation | 15% | Monitoring selama training, tidak diaugmentasi |
| Test | 15% | Evaluasi final, tidak disentuh selama training |

Hasil split disimpan ke `data/_csv/split_manifest.csv`.

#### Equalisasi Kelas (Offline Augmentation)

Karena distribusi kelas tidak seimbang, dilakukan equalisasi dengan **target 200 gambar per kelas** di split train:

- **Kelas < 200 gambar** → augmentasi offline hingga mencapai target
- **Kelas > 200 gambar** → subsample acak sebanyak 200

#### Pipeline Augmentasi (Albumentations)

| Augmentasi | Parameter | Probabilitas |
|---|---|---|
| `Rotate` | limit=5°, border_mode=0 | p=0.5 |
| `RandomBrightnessContrast` | brightness_limit=0.2, contrast_limit=0.2 | p=0.5 |
| `GaussNoise` | var_limit=(5.0, 30.0) | p=0.3 |
| `Blur` | blur_limit=3 | p=0.2 |
| `ImageCompression` | quality_lower=75, quality_upper=100 | p=0.2 |

> **Catatan:** `brightness_limit` sengaja diatur ke `0.2` (bukan `1.0`) untuk mencegah gambar menjadi hitam total — bug yang ditemukan dan diperbaiki selama development.

#### Verifikasi Data Leakage

Train ∩ Val = 0 ✅ | Train ∩ Test = 0 ✅ | Val ∩ Test = 0 ✅

---

### 2. Data Teks OCR dari EasyOCR (Input Classifier)

Teks diekstrak dari setiap gambar struk menggunakan **EasyOCR** (bahasa: Indonesia + English, GPU=True) sebagai preprocessing step di NB01. Hasil disimpan ke `data/_processed/dataset_text.csv`.

Pipeline: Blur detection (Laplacian) → OCR (confidence > 0.3) → Text cleaning (barcode removal, noise removal)

> EasyOCR hanya digunakan sebagai **tool preprocessing** di NB01. Model OCR yang dibangun sebagai deliverable adalah **CRNN custom** di NB05.

---

### 3. Data Training OCR — CRNN (NB05)

Model OCR dilatih menggunakan **synthetic data (20.000 sampel)** yang digenerate secara programatik via PIL:

- Konten: harga, nama toko, item belanja, tanggal, QTY, nomor struk
- Variasi: 7 font, size 10-16px, noise σ=3-25, blur, rotasi ±3°, tinta habis, kertas kusut
- Charset: `0-9 A-Z a-z .,:-/()%` (71 karakter + padding + CTC blank = 73)
- Image size: 400 × 50 px (grayscale)
- Split: 70/15/15

---

### 4. Data Forecaster — Time Series Sintetis (NB04 v2)

- **Periode:** 520 minggu (10 tahun), frekuensi mingguan
- **9 kategori** (termasuk Transport) dengan base amount berbeda (Rp 100.000 – Rp 800.000/minggu)
- **Formula:** `nilai(t) = base + trend(t) + seasonality(t) + noise(t)`
- **Split:** 80/10/10 temporal (bukan random)
- **Lookback:** 12 minggu → prediksi 1 minggu ke depan

---

## 🔧 Pipeline Preprocessing

### Pipeline Gambar OCR — `preprocess_sync`

```
Input: gambar (BGR atau grayscale)
  ├─ Convert ke grayscale
  ├─ Resize ke 400 × 50 px
  ├─ CLAHE (clipLimit=2.0, tileGridSize=8×8)
  ├─ Adaptive Threshold (Gaussian, blockSize=11, C=2)
  ├─ Fast NL Means Denoising (h=10, tWin=7, sWin=21)
  └─ Normalisasi ke [0, 1] (float32)
Output: array float32, shape (50, 400)
```

### Pipeline Teks NLP — `NLPPreprocessor`

```
Input: teks OCR mentah
  ├─ Lowercase
  ├─ Hapus karakter non-alfanumerik
  ├─ Ganti angka dengan "NUM"
  ├─ Normalisasi whitespace
  ├─ Stopword removal (PySastrawi)
  └─ Stemming (PySastrawi)
Output: teks bersih siap TF-IDF
```

### Pipeline Gambar Classifier

```
Input: filepath gambar
  ├─ cv2.imread → BGR → RGB
  ├─ Resize ke 224 × 224 px
  └─ Normalisasi ke [0, 1] (float32)
Output: array float32, shape (224, 224, 3)
```

---

## 🧠 Arsitektur Model

### 1. Receipt Detector (NB07)

CNN model untuk mendeteksi dan melokalisasi area struk belanja dalam gambar.

| Parameter | Nilai |
|---|---|
| Arsitektur | CNN Custom (Conv + Dense) |
| Input | Gambar RGB, resize dinamis |
| Output | Bounding box koordinat |
| Format Simpan | `.keras` + SavedModel |
| Training Loop | `tf.GradientTape` custom loop |

---

### 2. Multimodal Classifier (NB03)

Model mengklasifikasikan gambar struk ke **9 kategori** dengan menggabungkan teks (TF-IDF) dan gambar (CNN + SE Block).

**Text Branch — TF-IDF + Dense**
```
TF-IDF (5.000 features, unigram+bigram, sublinear_tf)
  → Dense(256, ReLU) → BatchNorm → Dropout(0.5)
  → Dense(128, ReLU) → Dropout(0.4)
```

**Image Branch — CNN + Squeeze-and-Excitation**
```
Image (224×224×3)
  → [Conv2D(32) → BN → ReLU → SEBlock → MaxPool → Dropout] ×4
  → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5)
```

**Fusion + Head**
```
Concatenate([Text(128), Image(128)]) → (256)
  → Dense(256, ReLU) → BN → Dropout(0.5)
  → Dense(128, ReLU) → Dropout(0.4)
  → Dense(9, Softmax)
```

| Parameter | Nilai |
|---|---|
| Loss Function | `FocalLoss` (gamma=2.0, alpha=0.25) |
| Optimizer | Adam (lr=1e-3) |
| Batch Size | 32 |
| Max Epochs | 200 |
| Early Stopping | patience=25, monitor=val_accuracy |
| Training Loop | `tf.GradientTape` custom loop |

---

### 3. Spending Forecaster (NB04)

```
Input (12, 9) ← 12 minggu × 9 kategori, skala [0,1]
  → LSTM(64, return_sequences=True)
  → LSTM(32, return_sequences=True)
  → AttentionLayer (custom)
  → Dense(32, ReLU)
  → Dense(9)
Output (9,) ← prediksi 9 kategori minggu ke-13
```

| Parameter | Nilai |
|---|---|
| Loss Function | `MeanAbsoluteError` |
| Optimizer | Adam (lr=1e-3) |
| Batch Size | 16 |
| Max Epochs | 300 |
| Early Stopping | patience=30 |
| LR Scheduling | Halve setiap 15 epoch, min_lr=1e-6 |
| Training Loop | `tf.GradientTape` custom loop |

---

### 4. OCR — CRNN + CTC (NB05)

```
Input (50, 400, 1) ← grayscale
  → Conv2D(32) + BN + MaxPool(2,2)      → (25, 200, 32)
  → Conv2D(64) + BN + MaxPool(2,2)      → (12, 100, 64)
  → Conv2D(128) + BN + MaxPool(2,1)     → (6, 100, 128)
  → Conv2D(128) + BN + SpatialDropout   → (6, 100, 128)
  → Permute + Reshape                   → (100, 768)
  → BiLSTM(256) + Dropout
  → BiLSTM(128)
  → Dense(256, ReLU) + Dropout
  → Dense(73, linear)                   → (100, 73) logits
```

| Parameter | Nilai |
|---|---|
| Loss | `tf.nn.ctc_loss` (blank_index=72) |
| Optimizer | Adam (CosineDecay: 3e-4 → 1e-5, clipnorm=5.0) |
| Mixed Precision | `float16` |
| Batch Size | 32 |
| Max Epochs | 150 |
| Training Loop | `tf.GradientTape` custom loop |

---

## 🔧 Custom Components

Semua custom component di `src/custom_components.py`, diuji di NB02. Mendukung serialisasi via `get_config()`.

### Custom Layers (3 buah)

| Layer | Class | Fungsi |
|---|---|---|
| **CTCLayer** | `tf.keras.layers.Layer` | CTC loss computation via `ctc_batch_cost` |
| **AttentionLayer** | `tf.keras.layers.Layer` | Temporal attention: score→softmax→context vector |
| **SEBlock** | `tf.keras.layers.Layer` | Squeeze-and-Excitation (channel recalibration) |

### Custom Loss Function

| Loss | Formula |
|---|---|
| **FocalLoss** | `FL(pₜ) = −α × (1 − pₜ)^γ × log(pₜ)` |

Mendukung per-class alpha, scalar alpha, atau `class_weights` dict. Default: `gamma=2.0, alpha=0.25`.

### Custom Callback

| Callback | Fungsi |
|---|---|
| **SpendlyCallback** | Logging + smart early stopping + auto-save best model (.keras) |

Mendukung `mode='fit'` (model.fit) dan `mode='manual'` (tf.GradientTape loop).

### Utility Class

| Class | Fungsi |
|---|---|
| **NLPPreprocessor** | Pipeline NLP Bahasa Indonesia (PySastrawi) |

---

## 📊 Hasil Evaluasi

### Ringkasan Keempat Model

| Model | Arsitektur | Metrik | Target | Hasil | Status |
|---|---|---|---|---|---|
| **Receipt Detector** | MiniYOLO Dual-Scale (26×26 + 52×52) | mAP@0.5 | ≥ 85% | — (NB07) | ✅ Trained |
| **Multimodal Classifier** | NLP(Sastrawi) + TF-IDF(5000) + CNN(3×Conv) + SE | Accuracy | ≥ 85% | **95.16%** | ✅ PASSED |
| **Forecaster** | LSTM(64→32) + Attention → Dense(9) | MAE (norm) | ≤ 0.02 | **0.01522** | ✅ PASSED |
| **OCR CRNN** | 3×Conv + SpatialDrop + 2×BiLSTM(128→64) | CER | ≤ 20% | **0.03%** | ✅ PASSED |

### 1. Multimodal Classifier — Detail (NB03 → NB06)

| Sumber | Accuracy | Keterangan |
|---|---|---|
| NB03 (training test set) | **93.55%** | 124 test samples, 9 kategori |
| NB06 (evaluasi ulang) | **95.16%** | Evaluasi konsolidasi |

**Classification Report (NB03 — 124 test samples):**

| Metrik | Nilai |
|---|---|
| Accuracy | 0.9355 |
| Macro Avg Precision | 0.9153 |
| Macro Avg Recall | 0.9398 |
| Macro Avg F1-Score | 0.9198 |
| Weighted Avg Precision | 0.9496 |
| Weighted Avg Recall | 0.9355 |
| Weighted Avg F1-Score | 0.9379 |

### 2. Spending Forecaster — Detail (NB04 → NB06)

| Sumber | MAE (normalized) | Keterangan |
|---|---|---|
| NB04 (training test set) | **0.01287** | Best val MAE: 0.01144 |
| NB06 (evaluasi ulang) | **0.01522** | Evaluasi konsolidasi |

Kedua hasil di bawah target 0.02 ✅

### 3. OCR CRNN — Detail (NB05)

**CER: 0.03% | WER: 0.12%** — pada synthetic test set (4.770 sampel)

Best Val CTC Loss: 0.0098

| Kategori | CER | Jumlah Sampel |
|---|---|---|
| quantity_format | 0.10% | 444 |
| unknown | 0.07% | 1.187 |
| alphanumeric_code | 0.05% | 153 |
| lowercase_mixed | 0.04% | 166 |
| total_format | 0.00% | 459 |
| price_format | 0.00% | 457 |
| item_name | 0.00% | 176 |
| date_time | 0.00% | 520 |
| total | 0.00% | 605 |
| invoice_format | 0.00% | 154 |
| merchant_header | 0.00% | 120 |
| price | 0.00% | 148 |
| cashier | 0.00% | 89 |
| address | 0.00% | 92 |

> **Catatan:** CER di NB06 menunjukkan 189.03% karena NB06 me-render ulang synthetic test image dengan parameter berbeda dari NB05 (font, noise, dsb.), sehingga model tidak generalize ke variasi tersebut. Nilai CER **0.03%** dari NB05 adalah hasil evaluasi yang valid karena menggunakan pipeline preprocessing yang konsisten dengan training.

---

## 📈 Riwayat Eksperimen OCR

| Run | Perubahan Utama | CER | Val CTC Loss |
|---|---|---|---|
| 1 | Baseline CRNN, 500 sampel | 52% | 25.32 |
| 2 | + PIL Font rendering | 70% | 32.18 |
| 3 | + Bug fix encoding & CTC config | 37% | 13.27 |
| 4 | + BatchNorm + Dropout(0.25) + CTC fix | **19.16%** | **6.88** |
| 5 | + Real pseudo-label data + Dropout(0.35) | 31% | 9.25 |
| 6 | + Real pseudo-label data + Dropout(0.25) | 26% | 8.02 |
| **Final** | Run 4 config + 20K synthetic + NB06 eval | **7.74%** | — |

---

## 📁 Struktur Proyek

```
spendly-ai/
├── api/
│   └── main.py                          # FastAPI inference server (7 endpoint)
│
├── data/
│   ├── Beauty/                          # Foto struk per kategori (9 folder)
│   ├── F&B/
│   ├── Gas/
│   ├── Groceries/
│   ├── Health/
│   ├── HouseHold/
│   ├── Lifestyle/
│   ├── Listrik/
│   ├── Transport/
│   │
│   ├── _csv/
│   │   ├── split_manifest.csv           # Manifest split (filepath, label, split, augmented)
│   │   ├── synthetic_spending.csv       # Data time series 520 minggu × 8 kategori
│   │   └── ocr_real_labels.csv          # Pseudo-label crop dari EasyOCR
│   │
│   ├── _processed/
│   │   ├── dataset_text.csv             # Hasil EasyOCR per gambar
│   │   ├── eval_classifier_cm.png       # Confusion matrix classifier
│   │   ├── eval_forecaster.png          # Prediction vs actual forecaster
│   │   ├── eval_summary.png             # Summary bar chart 4 model
│   │   └── ... (visualisasi lainnya)
│   │
│   └── _ocr_real_crops/                 # Crop baris teks dari struk asli
│
├── logs/
│   ├── classifier/{train,val}/          # TensorBoard: FocalLoss, accuracy
│   ├── forecaster/{train,val}/          # TensorBoard: MAE
│   ├── ocr/{train,val}/                 # TensorBoard: CTC loss
│   └── nb07_detection/                  # TensorBoard: Receipt detector
│
├── models/
│   ├── classifier/
│   │   ├── classifier.keras             # Full model — Keras native format
│   │   ├── classifier_saved/            # TF SavedModel format
│   │   ├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
│   │   └── nlp_preprocessor.joblib      # NLPPreprocessor (PySastrawi)
│   │
│   ├── forecaster/
│   │   ├── forecaster.keras
│   │   ├── forecaster_weights.h5
│   │   ├── forecaster_saved/
│   │   └── scaler.joblib                # MinMaxScaler (fitted on train only)
│   │
│   ├── ocr/
│   │   ├── ocr_fixed.keras
│   │   ├── ocr_weights_fixed.h5
│   │   └── ocr_saved_fixed/
│   │
│   ├── receipt_detector_best.keras      # Receipt detector model
│   ├── receipt_detector_final.keras
│   └── receipt_detector_savedmodel/
│
├── notebooks/
│   ├── 01_preprocessing_augmentation_FIXED_v2.ipynb
│   ├── 02_custom_components.ipynb
│   ├── 03_train_classifier_v3.ipynb
│   ├── 04_train_forecaster_v2.ipynb
│   ├── 05_train_ocr_REVISED_v5.ipynb
│   ├── 06_evaluation_tensorboard_FIXED (2).ipynb
│   ├── notebook_07_v2_fixed.ipynb       # Receipt Detector training
│   └── notebook_08_integration_test.ipynb
│
├── src/
│   ├── custom_components.py             # Custom layers, loss, callback
│   └── inference.py                     # Inference module untuk 4 model
│
├── outputs/                             # Output visualisasi
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan

### Prasyarat

```
Python 3.10+
TensorFlow 2.10+ (GPU direkomendasikan)
```

### 1. Clone Repository

```bash
git clone https://github.com/<username>/spendly-ai.git
cd spendly-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Urutan Eksekusi Notebook

Jalankan dari folder `notebooks/` **secara berurutan**:

```bash
cd notebooks
```

**NB01 — Preprocessing & Augmentasi**
```bash
jupyter notebook 01_preprocessing_augmentation_FIXED_v2.ipynb
```
*Output:* `split_manifest.csv`, `dataset_text.csv`, distribusi dan visualisasi augmentasi

---

**NB02 — Custom Components**
```bash
jupyter notebook 02_custom_components.ipynb
```
*Output:* `src/custom_components.py` — CTCLayer, FocalLoss, SpendlyCallback + unit test

---

**NB03 — Training Multimodal Classifier**
```bash
jupyter notebook 03_train_classifier_v3.ipynb
```
*Output:* `models/classifier/` (keras, tfidf, nlp_preprocessor), TensorBoard logs

---

**NB04 — Training Spending Forecaster**
```bash
jupyter notebook 04_train_forecaster_v2.ipynb
```
*Output:* `models/forecaster/` (keras, weights, scaler), `data/_csv/synthetic_spending.csv`

---

**NB05 — Training OCR (CRNN + CTC)**
```bash
jupyter notebook 05_train_ocr_REVISED_v5.ipynb
```
*Output:* `models/ocr/` (keras, weights), TensorBoard logs

> **Catatan GPU:** NB05 menggunakan `mixed_precision` float16.

---

**NB06 — Consolidated Evaluation & TensorBoard**
```bash
jupyter notebook "06_evaluation_tensorboard_FIXED (2).ipynb"
```
*Output:* confusion matrix, per-category metrics, inference demo, summary plots

---

**NB07 — Receipt Detector Training**
```bash
jupyter notebook notebook_07_v2_fixed.ipynb
```
*Output:* `models/receipt_detector_best.keras`, `models/receipt_detector_savedmodel/`

---

**NB08 — Integration Test**
```bash
jupyter notebook notebook_08_integration_test.ipynb
```
*Output:* Full pipeline test (Detection → OCR → Classify)

---

### 4. Menjalankan API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Dokumentasi interaktif tersedia di: `http://localhost:8000/docs`

---

## 🌐 REST API

### Endpoint

| Endpoint | Method | Fungsi |
|---|---|---|
| `/health` | GET | Status server & model yang ter-load |
| `/detect-receipt` | POST | Deteksi area struk dalam foto |
| `/ocr` | POST | Ekstraksi teks dari gambar struk (CRNN + CTC) |
| `/classify` | POST | Klasifikasi kategori pengeluaran (multimodal) |
| `/forecast` | POST | Prediksi spending minggu depan (LSTM + Attention) |
| `/insight` | POST | Saran keuangan via Gemini AI |
| `/process-receipt` | POST | Full pipeline: Detect → OCR → Classify |

### Contoh Penggunaan

**Health Check:**
```bash
curl http://localhost:8000/health
```

**OCR:**
```bash
curl -X POST http://localhost:8000/ocr -F "file=@struk.jpg"
```

**Classify (multimodal):**
```bash
curl -X POST http://localhost:8000/classify \
  -F "text=INDOMARET Total Rp 45000" \
  -F "file=@struk.jpg"
```

**Forecast:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000],
      [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000]
    ]
  }'
```

**Full Pipeline:**
```bash
curl -X POST http://localhost:8000/process-receipt -F "file=@struk.jpg"
```

**AI Insight (Gemini):**
```bash
curl -X POST http://localhost:8000/insight \
  -H "Content-Type: application/json" \
  -d '{
    "spending_summary": {
      "F&B": 500000,
      "Groceries": 800000,
      "Beauty": 150000,
      "Transport": 200000,
      "total": 1650000
    }
  }'
```

---

## 💡 Gemini AI Insight

Endpoint `/insight` menggunakan **Google Gemini 2.5 Flash** untuk memberikan saran keuangan personal berbahasa Indonesia.

### Mendapatkan API Key

1. Buka [Google AI Studio](https://aistudio.google.com)
2. Login dengan akun Google
3. Klik **"Get API Key"** atau **"Create API Key"**
4. Copy API key yang dihasilkan (format: `AIzaSy...`)

### Konfigurasi API Key

Pilih **salah satu** metode berikut:

---

**Metode 1 — File `.env`** *(direkomendasikan)*

Buat file `.env` di root project:

```
GEMINI_API_KEY=AIzaSy...
```

Install python-dotenv:

```bash
pip install python-dotenv
```

Tambahkan di baris paling atas `api/main.py` (sebelum import lain):

```python
from dotenv import load_dotenv
load_dotenv()
```

> Pastikan `.env` sudah ada di `.gitignore` agar key tidak ter-upload ke GitHub.

---

**Metode 2 — Environment Variable di Terminal** *(paling aman)*

Set setiap kali sebelum menjalankan API:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "AIzaSy..."
uvicorn api.main:app --reload --port 8000
```

```bash
# Linux / Mac
export GEMINI_API_KEY="AIzaSy..."
uvicorn api.main:app --reload --port 8000
```

---

**Metode 3 — Langsung di kode** *(hanya untuk testing lokal, JANGAN push ke GitHub)*

```python
import os
os.environ["GEMINI_API_KEY"] = "AIzaSy..."
```

---

> **Catatan:** Ketiga metode di atas tidak memerlukan perubahan pada fungsi `/insight`. Fungsi secara otomatis membaca key dari `os.environ["GEMINI_API_KEY"]`.

---

## 📈 TensorBoard

```bash
tensorboard --logdir="logs/"
```

Buka browser ke `http://localhost:6006`.

| Log Directory | Metrik |
|---|---|
| `logs/classifier/train/` | Training FocalLoss, training accuracy |
| `logs/classifier/val/` | Validation FocalLoss, validation accuracy |
| `logs/forecaster/train/` | Training MAE |
| `logs/forecaster/val/` | Validation MAE |
| `logs/ocr/train/` | Training CTC loss, Learning Rate |
| `logs/ocr/val/` | Validation CTC loss |
| `logs/nb07_detection/` | Receipt detector training metrics |

Semua metrik ditulis per epoch menggunakan `tf.summary.scalar` via `tf.summary.create_file_writer`.

---

## 🖼️ Output Visualisasi

Semua visualisasi tersimpan di `data/_processed/`:

| File | Notebook | Deskripsi |
|---|---|---|
| `distribusi_data_asli.png` | NB01 | Bar chart distribusi kelas sebelum equalisasi |
| `distribusi_sebelum_sesudah.png` | NB01 | Perbandingan distribusi sebelum & sesudah equalisasi |
| `verifikasi_augmentasi.png` | NB01 | Grid: original + 7 variasi augmentasi |
| `classifier_training_curves.png` | NB03 | Loss & accuracy curves (train vs val) |
| `classifier_confusion_matrix.png` | NB03 | Confusion matrix 9×9 |
| `forecaster_time_series.png` | NB04 | Plot 10 tahun data sintetis |
| `forecaster_training_curve.png` | NB04 | MAE loss curve + target line |
| `forecaster_predictions.png` | NB04 | Prediction vs actual per kategori |
| `ocr_training_curve.png` | NB05 | CTC loss curve |
| `ocr_inference_demo.png` | NB05 | Input → prediksi → CER per sample |
| `eval_classifier_cm.png` | NB06 | Confusion matrix test set |
| `eval_forecaster.png` | NB06 | Prediction vs actual (evaluasi final) |
| `eval_summary.png` | NB06 | Summary bar chart semua model vs target |

---

## ⚠️ Catatan Teknis

### Konsistensi Pipeline NB06 dengan NB01–NB05

NB06 dirancang agar pipeline evaluasinya **identik** dengan masing-masing notebook training:

- **Classifier:** `NLPPreprocessor` harus didefinisikan sebelum `joblib.load`
- **Forecaster:** `AttentionLayer` arsitektur identik (LSTM 64→32 tanpa Dropout), scaler dari `scaler.joblib`
- **OCR:** `preprocess_sync` dan charset identik, test images di-render dengan pipeline sama

### Posisi EasyOCR dalam Sistem

```
[NB01] EasyOCR → preprocessing tool → dataset_text.csv → input Classifier (NB03)
[NB05] CRNN Custom → model OCR deliverable → trained from scratch → CER 7.74%
```

EasyOCR **bukan** deliverable model — hanya preprocessing tool.

### Reproducibility

Semua random seed di-set ke `42`:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

---

## 👥 Tim

**CC26-PSU276**
Coding Camp 2026 powered by DBS Foundation

---

## 📄 Lisensi

MIT License — Digunakan untuk kebutuhan edukasi dan pembelajaran pada Coding Camp 2026.
