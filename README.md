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
  - Warna: Histogram HSV, rata-rata channel warna
  - Bentuk: Kontur, aspect ratio, extent, compactness, Hu Moments
  - Tekstur: GLCM (Haralick), Local Binary Pattern
- Klasifikasi multi-kelas
- Visualisasi hasil ekstraksi dan klasifikasi KNN

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
- `fitur_warna.npz`: Menyimpan fitur-fitur warna dari setiap gambar
- `fitur_bentuk.csv`: Menyimpan fitur-fitur bentuk dari setiap gambar
- `fitur_tekstur.npz`: Menyimpan fitur-fitur tekstur dari setiap gambar

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
python kode/ekstraksi_warna.py;
python kode/ekstraksi_bentuk.py;
python kode/ekstraksi_tekstur.py
```

## Pembagian Tugas

- **152023070 Reeyhan Arif Saputra**
  - Pengumpulan dataset awal
  - Preprocessing citra dasar
  - Ekstraksi fitur bentuk
  - Ekstraksi fitur warna
  - Ekstraksi fitur tekstur
  - Implementasi klasifikasi dasar
  - Pembuatan video demo

- **152023061 M. Bakti Komara R. P.**
  - Pengembangan dan penambahan model klasifikasi
  - Pembuatan fitur live feed
  - Implementasi variasi preprocessing
  - Pembuatan video demo

- **152023055 Muhammad Kevin**
  - Penambahan dataset baru (60 Biji)
  - Implementasi augmentasi data
  - Penyusunan laporan teknis lengkap
  - Pengujian dan validasi model
  - Lead pembuatan video demo

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
