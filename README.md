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

Instalasi dependencies menggunakan command berikut:

```
pip install -r requirements.txt
```

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

- **152023055 Muhammad Kevin**
  - Pengumpulan dataset
  - Preprocessing citra
  - Ekstraksi fitur bentuk

- **152023061 M. Bakti Komara R. P.**
  - Ekstraksi fitur warna
  - Ekstraksi fitur tekstur
  - Dokumentasi laporan

- **152023070 Reeyhan Arif Saputra**
  - Implementasi klasifikasi
  - Evaluasi performa
  - Video demo dan laporan klasifikasi

## Lisensi

MIT License