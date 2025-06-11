# C8 - Klasifikasi Sampah Domestik Berbasis Pengolahan Citra Digital

Sistem klasifikasi sampah domestik menggunakan ekstraksi fitur warna, bentuk, dan tekstur melalui pengolahan citra digital. Proyek ini menggunakan kombinasi fitur HSV, kontur, dan tekstur untuk mengklasifikasikan sampah ke dalam tiga kategori.

## Deskripsi

Proyek ini bertujuan untuk mengklasifikasikan sampah domestik ke dalam 3 kategori utama:
- **Organik** (kulit buah, daun) - berbasis analisis bentuk
- **Plastik** (botol, kantong plastik) - berbasis analisis tekstur
- **Kertas** (kardus, kertas bekas) - berbasis analisis warna

## Fitur Utama

- Preprocessing citra (resizing, normalisasi, konversi warna)
- Ekstraksi fitur:
  - Warna: Histogram HSV, rata-rata channel warna
  - Bentuk: Kontur, aspect ratio, extent, compactness, Hu Moments
  - Tekstur: GLCM (Haralick), Local Binary Pattern
- Klasifikasi multi-kelas
- Visualisasi hasil ekstraksi dan klasifikasi KNN

## Alur Pemrosesan

### Ekstraksi Fitur Bentuk
```
Citra RGB → Grayscale → Biner (Otsu) → Deteksi Kontur → Fitur Geometri
                                                         - Luas
                                                         - Rasio Aspek
                                                         - Jumlah Sisi
```

### Ekstraksi Fitur Warna
```
Citra RGB → HSV → Histogram HSV (512 bin) → Fitur Warna
        └─→ Mean RGB ────────────────────┘
```

### Ekstraksi Fitur Tekstur
```
Citra RGB → Grayscale → GLCM → Properti Haralick
                              - Contrast
                              - Dissimilarity
                              - Homogeneity
                              - Energy
                              - Correlation
```

## Teknologi yang Digunakan

- Python 3.12
- OpenCV (preprocessing & ekstraksi bentuk)
- NumPy (operasi numerik)
- Mahotas (ekstraksi tekstur)
- scikit-image (Local Binary Pattern)
- scikit-learn (klasifikasi)
- Matplotlib (visualisasi)

## Struktur Dataset

```
citra/
├── asli/                   # Master copies dari semua gambar
│   ├── kertas_samples/     # Gambar sampah kertas asli
│   ├── organik_samples/    # Gambar sampah organik asli
│   └── plastik_samples/    # Gambar sampah plastik asli
├── training/               # Dataset pelatihan
│   ├── kertas/            # Gambar kertas
│   ├── organik/           # Gambar organik
│   └── plastik/           # Gambar plastik
└── testing/               # Dataset pengujian
    ├── kertas/            # Contoh: kertas.png
    ├── organik/           # Contoh: pisang.png
    └── plastik/           # Contoh: botol.png
```

## Hasil Ekstraksi Fitur

Hasil ekstraksi fitur disimpan dalam folder `hasil_ekstraksi/` dalam format yang terstruktur:
- `fitur_warna.csv` & `fitur_warna.npz`: Menyimpan fitur-fitur warna dari setiap gambar
- `fitur_bentuk.csv`: Menyimpan fitur-fitur bentuk dari setiap gambar
- `fitur_tekstur.npz`: Menyimpan fitur-fitur tekstur dari setiap gambar

Visualisasi hasil ekstraksi fitur tersimpan dalam subfolder:
- `visualisasi_warna/`: Hasil visualisasi ekstraksi fitur warna
- `visualisasi_bentuk/`: Hasil visualisasi ekstraksi fitur bentuk
- `visualisasi_tekstur/`: Hasil visualisasi ekstraksi fitur tekstur

## Cara Penggunaan

1. Clone repository
```bash
git clone https://github.com/Vortexaint/PCD-RedOnion.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Jalankan program ekstraksi fitur
```bash
python .\kode\augmentasi_data.py
python .\kode\ekstraksi_warna.py
python .\kode\ekstraksi_tekstur.py
python .\kode\ekstraksi_bentuk.py
python .\kode\ekstraksi.py
```

## Pembagian Tugas

- **152023070 Reeyhan Arif Saputra**
  - Dataset dan Struktur:
    - Pengumpulan dataset
    - Pengaturan struktur folder proyek
  - Preprocessing dan Ekstraksi:
    - Preprocessing citra dasar
    - Ekstraksi fitur bentuk (`ekstraksi_bentuk.py`)
    - Ekstraksi fitur warna (`ekstraksi_warna.py`)
    - Ekstraksi fitur tekstur (`ekstraksi_tekstur.py`)
  - Model dan Validasi:
    - Implementasi klasifikasi dasar
    - Validasi model dan metrik (`validasi_model.py`)
    - Pengembangan confusion matrix
  - Augmentasi dan Visualisasi:
    - Implementasi augmentasi data (`augmentasi_data.py`)
    - Implementasi visualisasi preprocessing
  - Integrasi:
    - Implementasi integrasi antar-modul
    - Pembuatan video demo

- **152023061 M. Bakti Komara R. P.**
  - Pengembangan model klasifikasi multi-kelas
  - Implementasi kombinasi fitur (`ekstraksi_kombinasi.py`)
  - Implementasi preprocessing tambahan
  - Pembuatan video demo

- **152023055 Muhammad Kevin**
  - Implementasi augmentasi data (`augmentasi_data.py`)
  - Penyusunan dokumentasi teknis
  - Pengujian performa model
  - Lead pembuatan video demo

# Akurasi

Model KNN:
Fitur Bentuk: 53.13% (0.53125)
Fitur Tekstur: 83.33% (0.8333)
Fitur Warna: 95.83% (0.9583)
Model SVM:
Fitur Bentuk: 26.04% (0.2604)
Fitur Tekstur: 70.83% (0.7083)
Fitur Warna: 97.92% (0.9792)
Perbandingan:

Fitur Bentuk:

KNN lebih baik (53.13%)
SVM lebih rendah (26.04%)
Fitur Tekstur:

KNN lebih baik (83.33%)
SVM cukup baik (70.83%)
Fitur Warna:

SVM sedikit lebih baik (97.92%)
KNN juga sangat baik (95.83%)

## Lisensi

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   Copyright 2024 Red Onion

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
