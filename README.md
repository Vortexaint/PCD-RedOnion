# Klasifikasi Sampah Domestik Berbasis Pengolahan Citra Digital

Sistem klasifikasi sampah domestik menggunakan ekstraksi fitur warna, bentuk, dan tekstur melalui pengolahan citra digital. Proyek ini menggunakan kombinasi fitur HSV, kontur, dan tekstur untuk mengklasifikasikan sampah ke dalam tiga kategori.

## Deskripsi

Proyek ini bertujuan untuk mengklasifikasikan sampah domestik ke dalam 3 kategori utama:
- **Organik** (kulit buah, daun) - berbasis analisis bentuk
- **Plastik** (botol, kantong plastik) - berbasis analisis tekstur
- **Kertas** (kardus, kertas bekas) - berbasis analisis warna

## Fitur Utama

- Preprocessing citra (resizing, normalisasi, konversi warna)
- Ekstraksi fitur:
  - Warna: Histogram HSV, rata-rata channel warna (96 dimensi)
  - Bentuk: Kontur, aspect ratio, extent, compactness, Hu Moments (10 dimensi)
  - Tekstur: GLCM (Haralick), Local Binary Pattern (13 + 26 dimensi)
- Klasifikasi multi-kelas menggunakan Random Forest
- Visualisasi hasil ekstraksi dan klasifikasi

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
├── training/               # Dataset pelatihan (70%)
│   ├── kertas/            # 8 gambar kertas
│   ├── organik/           # 4 gambar organik
│   └── plastik/           # 6 gambar plastik
└── testing/               # Dataset pengujian (30%)
    ├── kertas/            # 3 gambar kertas
    ├── organik/           # 2 gambar organik
    └── plastik/           # 3 gambar plastik
```

## Hasil Ekstraksi Fitur

Setiap gambar menghasilkan 3 file fitur di folder `hasil_ekstraksi/`:
- `*_color_features.npy`: 96 fitur warna (histogram HSV + rata-rata)
- `*_shape_features.npy`: 10 fitur bentuk (metrics kontur + Hu moments)
- `*_texture_features.npy`: 39 fitur tekstur (GLCM + LBP)

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
python kode/ekstraksi_warna.py
python kode/ekstraksi_bentuk.py
python kode/ekstraksi_tekstur.py
```

4. Jalankan klasifikasi
```bash
python kode/klasifikasi.py
```

## Pembagian Tugas

- **152023070 Reeyhan Arif Saputra**
  - Pengumpulan dataset awal
  - Preprocessing citra dasar
  - Ekstraksi fitur bentuk
  - Ekstraksi fitur warna
  - Ekstraksi fitur tekstur
  - Implementasi klasifikasi dasar dengan Random Forest

- **152023061 M. Bakti Komara R. P.**
  - Pengembangan model klasifikasi tambahan (KNN dan SVM)
  - Implementasi variasi preprocessing (RGB→CMYK, filtering)
  - Penambahan dataset baru (65% dari 34 gambar yang dibutuhkan)
  - Penyusunan laporan teknis lengkap
  - Konsultasi terkait penggunaan library dengan dosen

- **152023055 Muhammad Kevin**
  - Penambahan dataset baru (35% dari 34 gambar yang dibutuhkan)
  - Implementasi augmentasi data (rotasi, flip, variasi pencahayaan)
  - Pembuatan video demo
  - Implementasi evaluasi performa (akurasi dan confusion matrix)

## TO-DO LIST
Masalah Penggunaan Library
- Projek saat ini masih menggunakan beberapa library yang tidak diizinkan, yaitu:
  - matplotlib (digunakan untuk visualisasi)
  - scikit-learn (digunakan untuk klasifikasi)
  - pathlib (jika digunakan dalam kode)

Solusi: Hapus atau ganti semua library tersebut. Hanya library berikut yang diperbolehkan:
```
cv2, skimage, PIL, NumPy, mahotas
```
Atau konsultasi dengan Bapak Rizka Milandga Milenio
https://id.linkedin.com/in/milandga-milenio-462937182

Persyaratan Dataset
- Tugas mengharuskan penggunaan minimal 60 citra RGB. Dataset saat ini:
  - Training: 8 + 4 + 6 = 18 citra
  - Testing: 3 + 2 + 3 = 8 citra
  - Total: 26 citra

Solusi: Tambahkan minimal 34 citra lagi agar total mencapai minimal 60.

Model Klasifikasi
- Saat ini hanya menggunakan Random Forest.

Solusi: Tambahkan 2 model klasifikasi (contoh: KNN dan SVM).

Deliverables yang Masih Kurang
- Video demo
- Laporan lengkap (bukan hanya README), memuat:
  - Penjelasan preprocessing secara detail
  - Algoritma ekstraksi fitur
  - Algoritma klasifikasi
  - Hasil evaluasi performa klasifikasi

Evaluasi Kinerja
- Tugas meminta penggunaan akurasi (accuracy) sebagai matriks evaluasi.
- Meskipun scikit-learn tidak diizinkan secara umum, penggunaan untuk evaluasi diperbolehkan sesuai instruksi tugas.

Solusi: Tampilkan akurasi dan confusion matrix menggunakan scikit-learn.

Upaya Tambahan & Kreativitas
- Untuk nilai tambahan, tambahkan variasi preprocessing:
  - ✅ Sudah mencoba RGB → HSV
  - Bisa ditambahkan: RGB → CMYK
  - Tambah variasi kondisi citra:
  - Sudut pengambilan gambar berbeda
  - Pencahayaan terang/redup
  - Citra dengan noise/kabut
  - Tambahkan variasi teknik preprocessing:
    - Filtering (median, Gaussian, dsb)
    - Augmentasi (rotasi, flip, dsb)

Prioritas Paling Mendesak
- Tambah dataset menjadi minimal 60 citra
- Tambah minimal 1 model klasifikasi tambahan
- Buat video demo dan laporan lengkap
- Hapus/ganti semua library yang tidak diperbolehkan/Konsultasi

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
