# 🏨 Hotel Booking Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Proyek ini bertujuan untuk membangun **model machine learning** yang dapat memprediksi
apakah suatu reservasi hotel berpotensi **dibatalkan (cancel)** atau **tidak**.

Prediksi ini penting bagi industri perhotelan karena pembatalan reservasi secara langsung
berdampak pada:
- Ketidakpastian okupansi
- Risiko *lost revenue*
- Strategi *overbooking* dan *revenue management*

Model dikembangkan menggunakan pendekatan **supervised classification**
dengan fokus utama pada **minimasi false negative**.

---

## 📂 Repository Outline
```
1. README.md                 - Dokumentasi dan penjelasan project
2. EDA_dan_Modeling.ipynb    - Notebook EDA, feature engineering, modeling, dan evaluasi
3. Inference.ipynb           - Notebook untuk uji coba inference menggunakan data baru
4. dataset/
   ├── hotel_bookings.csv    - Dataset asli
5. deployment/
   ├── src/
   │   ├── streamlit.py      - Main Streamlit app (UI routing)
   │   ├── eda.py            - EDA visualization untuk deployment
   │   ├── predict.py        - Logic inference dan prediksi harga
   │   ├── model_rf_baseline.pkl
   │   ├── common_kecamatan.pkl
   │   └── final_features.pkl
   ├── requirements.txt      - Daftar dependency untuk deployment
6. url.txt                   - URL model dan deployment

```
---

## 🎯 Problem Statement

Permasalahan utama yang dihadapi industri perhotelan:

1. Tingginya pembatalan reservasi menyebabkan ketidakpastian permintaan.
2. Forecasting okupansi menjadi tidak akurat.
3. Strategi overbooking menjadi berisiko.
4. Diperlukan sistem prediktif untuk mendeteksi reservasi berisiko cancel sejak awal.

---

## 🧠 Objective

Membangun dan membandingkan beberapa model klasifikasi untuk memprediksi
pembatalan reservasi hotel.

Model yang diuji:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost

**Metrik evaluasi utama:**
- **Recall** → memaksimalkan deteksi pembatalan
- **ROC-AUC** → kemampuan pemisahan kelas secara keseluruhan

---

## 👥 Target Users

- Manajer hotel dan staf operasional
- Tim *revenue management*
- Analis data internal hotel

---

## 📊 Dataset

- **Sumber**: [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Ukuran**: 119.390 baris, 32 kolom
- **Target**: `is_canceled`
- **Missing values**: `country`, `children`, `agent`, `company`

Kolom yang berpotensi menyebabkan *data leakage*
(`reservation_status`, `reservation_status_date`)
tidak digunakan dalam modeling.

---

## ⚙️ Methodology

Tahapan utama proyek:

1. Exploratory Data Analysis (EDA)
2. Handling missing values & cardinality
3. Feature engineering & preprocessing (pipeline)
4. Baseline modeling
5. Stratified K-Fold Cross Validation
6. Hyperparameter tuning
7. Model evaluation & selection
8. Inference & deployment

Berdasarkan evaluasi, **Random Forest baseline**
dipilih sebagai **model final** karena memberikan
recall tertinggi dan performa yang paling stabil.

---

## 🏆 Model Performance (Final)

| Metric | Train | Test |
|------|------|------|
| Recall | ~0.99 | ~0.65 |
| ROC-AUC | ~1.00 | ~0.90 |

Model menunjukkan kemampuan pemeringkatan yang sangat baik,
dengan trade-off overfitting yang masih dapat diterima
sesuai tujuan bisnis (recall-oriented).

---

## 🔍 Example Inference Output

Contoh hasil prediksi pada data baru:

| Case | Predicted Class | Probability Cancel | Interpretation |
|----|----------------|--------------------|----------------|
| 1 | 0 | 0.405 | Tidak Akan Cancel |
| 2 | 0 | 0.150 | Tidak Akan Cancel |

> Interpretasi:
> - Kedua reservasi diprediksi **tidak dibatalkan**
> - Case pertama masih memiliki risiko moderat
> - Case kedua sangat kecil kemungkinan cancel

---

## 🧪 Tech Stack

**Programming Language**
- Python 3.9+

**Libraries**
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- phik
- pickle

---

## 🚀 Model Usage

Jika ingin langsung menggunakan model tanpa training ulang:

1. Unduh file `best_rf_model.pkl`
2. Load model menggunakan `pickle`
3. Gunakan notebook `Inference.ipynb` sebagai referensi inferensi

Model menerima data reservasi hotel dan menghasilkan:
- Prediksi cancel / tidak cancel
- Probabilitas pembatalan

---