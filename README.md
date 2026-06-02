# Face Alignment with dlib

Proyek ini melakukan face alignment dengan `dlib` dan `OpenCV`. Program membaca semua gambar dari folder `image`, mendeteksi wajah, mengambil landmark mata, lalu menampilkan gambar asli dan hasil yang sudah diluruskan secara berdampingan.

## Fitur

- Membaca semua gambar di folder `image/`
- Deteksi wajah dengan `dlib`
- Landmark wajah 68 titik
- Perhitungan sudut dari posisi kedua mata
- Rotasi gambar agar mata sejajar horizontal
- Tampilan perbandingan asli vs lurus tanpa menyimpan file output baru

## Prasyarat

- Python 3.x
- `opencv-python`
- `numpy`
- `dlib-bin`
- File model `shape_predictor_68_face_landmarks.dat`

## Struktur Folder

- `face_alignment_dlib.py` - skrip utama
- `image/` - folder gambar input
- `models/` - folder file model dlib
- `requirements.txt` - daftar dependensi

## Cara Menjalankan

1. Buat dan aktifkan virtual environment.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Pastikan file `shape_predictor_68_face_landmarks.dat` berada di folder `models/`.
4. Jalankan program:
   ```bash
   python face_alignment_dlib.py
   ```

## Alur Program

1. Ambil semua gambar dari folder `image/`.
2. Cari wajah terbesar pada setiap gambar.
3. Ambil landmark 68 titik dari wajah.
4. Hitung pusat mata kiri dan kanan.
5. Hitung sudut kemiringan wajah.
6. Rotasi gambar agar mata sejajar.
7. Tampilkan hasil asli dan hasil lurus secara berdampingan.

## Troubleshooting

- Jika `cv2` atau `dlib` error di Windows, pastikan Anda memakai virtual environment yang benar.
- Jika instalasi `dlib` dari source gagal, gunakan `dlib-bin` seperti di `requirements.txt`.
- Jika model belum ditemukan, pastikan file `shape_predictor_68_face_landmarks.dat` sudah disimpan di folder `models/`.