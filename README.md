# Flight Price Prediction

Proyek machine learning untuk memprediksi harga tiket pesawat maskapai India berdasarkan data penerbangan. Tiga algoritma regresi dibandingkan: **Ridge**, **Random Forest**, dan **XGBoost**, masing-masing dengan *hyperparameter tuning* di dalam pipeline preprocessing.

## Ringkasan

- **Data:** 300.153 baris, 11 kolom, tanpa missing value dan tanpa duplikat
- **Metrik utama:** R², RMSE, MAE, dan CV RMSE (3-fold)
- **Model terbaik:** XGBoost dengan R² 0,988 dan RMSE 2.503

## Dataset

[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) dari Kaggle (`Clean_Dataset.csv`), berisi atribut penerbangan:

| Kolom | Keterangan |
|---|---|
| `airline` | Maskapai |
| `flight` | Kode penerbangan |
| `source_city`, `destination_city` | Kota asal dan tujuan |
| `departure_time`, `arrival_time` | Waktu keberangkatan dan kedatangan |
| `stops` | Jumlah transit (`zero`, `one`, `two_or_more`) |
| `class` | Kelas tiket (Economy / Business) |
| `duration` | Durasi penerbangan (jam) |
| `days_left` | Jumlah hari antara pembelian tiket dan keberangkatan |
| `price` | Harga tiket (target) |

## Alur Kerja

1. **Data understanding:** `head`, `info`, `describe`, cek missing value dan duplikat.
2. **EDA:**
   - Histogram dan skewness fitur numerik (`price` 1,06; `duration` 0,60; `days_left` -0,04).
   - Boxplot untuk melihat outlier dan countplot fitur kategorikal (kolom `flight` hanya menampilkan 20 terbanyak).
   - Heatmap korelasi fitur numerik: tidak ada korelasi yang tinggi antar fitur.
   - Menjawab 5 *research questions* (lihat bagian Temuan EDA).
3. **Data cleaning:** kolom `flight` dihapus karena kardinalitasnya tinggi (1.561 nilai unik).
4. **Split data:** 80:20 dengan `random_state=42`.
5. **Preprocessing (di dalam pipeline):**
   - `OrdinalEncoder` untuk `stops` (zero < one < two_or_more) dan `class` (Economy < Business).
   - `OneHotEncoder` (`handle_unknown='ignore'`) untuk `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`.
   - `StandardScaler` untuk `duration` dan `days_left`, hanya untuk Ridge. Random Forest dan XGBoost memakai fitur numerik apa adanya (*passthrough*).
6. **Modeling dan tuning:** Ridge, Random Forest, dan XGBoost dengan `RandomizedSearchCV` (`n_iter=10`, `cv=3`, scoring `neg_root_mean_squared_error`).
7. **Evaluasi:** MAE, RMSE, R² pada data test, CV RMSE dari data training, 10 fitur terpenting, dan plot nilai aktual vs prediksi.
8. **Simpan model:** XGBoost terbaik disimpan ke `xgb_flight_price.joblib`.

Preprocessing berada di dalam `Pipeline`, sehingga di-*fit* hanya pada data training di setiap fold.

## Temuan EDA

- **Maskapai:** Vistara dan Air India memiliki harga jauh lebih mahal dibandingkan 4 maskapai lainnya.
- **Jumlah stop:** penerbangan 1 stop lebih mahal dibandingkan zero stop maupun two or more stop.
- **Days left:** pengaruhnya berbeda antar maskapai dan kelas. Harga tiket Business relatif stabil mulai sekitar 11 hari sebelum keberangkatan.
- **Waktu keberangkatan dan kedatangan:** memengaruhi harga, tetapi pengaruhnya tidak konsisten antar kombinasi.
- **Rute:** harga dipengaruhi kota asal dan tujuan. Pada Economy, rute termurah Mumbai-Hyderabad (5.775) dan termahal Kolkata-Chennai (8.012). Pada Business, rute termahal Kolkata-Bangalore (58.681).
- **Kelas tiket:** Business jauh lebih mahal dibandingkan Economy.

## Hasil

### Data Test

| Model | CV RMSE | MAE | RMSE | R² |
|---|---|---|---|---|
| **XGBoost** | 2585.38 | 1290.60 | 2503.39 | 0.9878 |
| Random Forest | 2716.15 | 1095.91 | 2688.54 | 0.9860 |
| Ridge | 6809.83 | 4500.76 | 6814.94 | 0.9099 |

XGBoost dipilih sebagai model terbaik karena RMSE, R², dan CV RMSE terbaik. Random Forest memiliki MAE terendah.

**Hyperparameter terbaik XGBoost:** `n_estimators=391`, `max_depth=8`, `learning_rate≈0.17`, `min_child_weight=2`, `subsample≈1.0`, `colsample_bytree≈0.99`.

**Fitur terpenting (XGBoost):** `class` mendominasi (importance sekitar 0,78), diikuti `airline` (Air India dan Vistara) dan `duration`.

## Keterbatasan

- Evaluasi memakai satu kali split train/test (80:20).
- Pencarian hyperparameter terbatas (10 iterasi, 3-fold CV).
- Target `price` tidak ditransformasi meskipun distribusinya miring dan harga Economy-Business sangat berbeda.
- Kolom `flight` tidak dipakai, sehingga informasi per nomor penerbangan hilang.

## Rencana Pengembangan

- Memperbanyak iterasi tuning atau memakai metode pencarian lain.
- Evaluasi performa model secara terpisah untuk kelas Economy dan Business.
- Transformasi target (misalnya log) dan feature engineering (misalnya kombinasi kota asal-tujuan).
- Menambah model pembanding (LightGBM, CatBoost, Gradient Boosting).

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, XGBoost, joblib
