# Face Alignment with dlib

Program ini membaca semua gambar di folder `image`, meluruskan wajah berdasarkan posisi kedua mata, lalu menyimpan hasilnya ke folder `aligned_images`.

## File penting

- `face_alignment_dlib.py`
- `requirements.txt`
- `models/shape_predictor_68_face_landmarks.dat`

## Cara menjalankan

1. Buat dan aktifkan virtual environment.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan program:
   ```bash
   python face_alignment_dlib.py
   ```

## Struktur folder

- `image/` untuk gambar input
- `models/` untuk file landmark dlib
- `aligned_images/` untuk hasil output

## Troubleshooting singkat

- Jika `dlib` atau `cv2` error di Windows, pastikan Anda memakai environment yang benar.
- Jika `dlib` gagal dipasang dari source, gunakan `dlib-bin` seperti di `requirements.txt`.
- Jika model belum ada, unduh `shape_predictor_68_face_landmarks.dat` lalu simpan ke folder `models`.