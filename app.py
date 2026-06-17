import math
from pathlib import Path

import cv2
import dlib
import numpy as np
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Face Alignment with dlib",
    page_icon="🧑‍💻",
    layout="wide"
)

# Konstanta Indeks Mata pada 68-Landmarks
LEFT_EYE = range(36, 42)
RIGHT_EYE = range(42, 48)

# ==========================================
# 2. FUNGSI PENDUKUNG (LOAD MODEL)
# ==========================================
@st.cache_resource
def load_models():
    """
    Memuat model dlib. Menggunakan @st.cache_resource agar 
    model hanya dimuat sekali dan tidak membebani memori setiap kali ada interaksi.
    """
    # Cari file model di folder saat ini atau subfolder 'models'
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "models" / "shape_predictor_68_face_landmarks.dat",
        current_dir / "shape_predictor_68_face_landmarks.dat",
    ]
    
    predictor_path = None
    for path in candidates:
        if path.exists():
            predictor_path = path
            break
            
    if predictor_path is None:
        raise FileNotFoundError(
            "File shape_predictor_68_face_landmarks.dat tidak ditemukan. "
            "Pastikan file tersebut ada di direktori project."
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    return detector, predictor

# ==========================================
# 3. FUNGSI INTI PEMROSESAN CITRA
# ==========================================
def get_scale_factor(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        return max_width / w
    return 1.0

def largest_face(detector, gray_image, max_width=800):
    """Mendeteksi seluruh wajah dan mengembalikan wajah dengan ukuran bounding box terbesar.
    Menerapkan downscaling otomatis untuk mempercepat deteksi pada gambar beresolusi tinggi.
    """
    scale = get_scale_factor(gray_image, max_width)
    if scale < 1.0:
        h, w = gray_image.shape[:2]
        small_gray = cv2.resize(gray_image, (int(w*scale), int(h*scale)))
    else:
        small_gray = gray_image
        
    faces = detector(small_gray, 1)
    if not faces:
        return None
        
    best_face = max(faces, key=lambda face: face.width() * face.height())
    
    if scale < 1.0:
        # Kembalikan koordinat bounding box ke skala aslinya
        return dlib.rectangle(
            int(best_face.left() / scale),
            int(best_face.top() / scale),
            int(best_face.right() / scale),
            int(best_face.bottom() / scale)
        )
    return best_face

def landmarks_68(predictor, gray_image, face):
    """Mengambil 68 titik landmark dari wajah yang terdeteksi."""
    shape = predictor(gray_image, face)
    points = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        points[i] = (shape.part(i).x, shape.part(i).y)
    return points

def eye_center(points, eye_indexes):
    """Menghitung titik tengah rata-rata (centroid) dari sekumpulan titik mata."""
    return points[list(eye_indexes)].mean(axis=0)

def rotate_to_align(image, left_eye, right_eye, points=None):
    """Merotasikan gambar agar posisi mata kiri dan kanan sejajar secara horizontal."""
    # Hitung selisih X dan Y antara mata kanan dan kiri
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Hitung sudut kemiringan dalam derajat
    angle = math.degrees(math.atan2(dy, dx))
    
    # Titik pusat rotasi berada di tengah-tengah antara kedua mata
    center = tuple(((left_eye + right_eye) / 2).tolist())

    # Lakukan Affine Transformation (Rotasi)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_img = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE # Mencegah tepian gambar menjadi hitam tajam
    )
    
    # Jika points (landmark) diberikan, rotasikan juga semua titiknya
    aligned_points = None
    if points is not None:
        aligned_points = np.zeros_like(points)
        for i, (x, y) in enumerate(points):
            aligned_points[i, 0] = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
            aligned_points[i, 1] = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
            
    return aligned_img, aligned_points

def draw_landmarks(image, points):
    """Menggambar 68 titik landmark pada gambar."""
    img_copy = image.copy()
    for (x, y) in points:
        cv2.circle(img_copy, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img_copy

# ==========================================
# 4. ANTARMUKA PENGGUNA (UI) DENGAN STREAMLIT
# ==========================================
def main():
    st.title("🧑‍💻 Face Alignment with dlib")
    st.markdown("""
    Aplikasi web ini menggunakan algoritma **Face Alignment** untuk meluruskan posisi kepala
    berdasarkan titik mata menggunakan model 68-landmarks dari `dlib`.
    """)

    # --- Bagian Memuat Model ---
    try:
        with st.spinner("Memuat model deteksi wajah dlib..."):
            detector, predictor = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop() # Hentikan aplikasi jika model tidak ditemukan

    # --- Bagian Sumber Gambar ---
    st.markdown("### 📁 Sumber Gambar")
    source_option = st.radio("Pilih sumber gambar:", ["Pilih dari Contoh (Folder 'image')", "Unggah Gambar Sendiri (Upload)"], horizontal=True)
    
    image_bgr = None
    
    if source_option == "Pilih dari Contoh (Folder 'image')":
        image_dir = Path("image")
        if image_dir.exists():
            image_files = [f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            if image_files:
                selected_image = st.selectbox("Pilih Gambar Contoh:", image_files)
                image_path = image_dir / selected_image
                # Membaca gambar dari path
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    st.error("Gagal membaca gambar dari folder.")
                    st.stop()
            else:
                st.warning("Tidak ada file gambar di dalam folder 'image'.")
        else:
            st.warning("Folder 'image' tidak ditemukan.")
            
    else:
        # --- Bagian Upload Gambar ---
        uploaded_file = st.file_uploader(
            "Unggah gambar wajah (JPG, JPEG, PNG, WEBP)", 
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file is not None:
            # Konversi file unggahan menjadi array byte yang bisa dibaca OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image_bgr is None:
                st.error("Gagal membaca gambar. Silakan coba file lain atau format lain.")
                st.stop()

    if image_bgr is not None:

        # Catatan: OpenCV menggunakan format warna BGR, sedangkan Streamlit/Browser menggunakan RGB.
        # Oleh karena itu, kita harus mengkonversi warnanya sebelum ditampilkan.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Buat layout 2 kolom (Kiri untuk Asli, Kanan untuk Lurus)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image_rgb, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("⚙️ Pengaturan Pemrosesan")
        
        show_landmarks = st.checkbox("🟢 Tampilkan Titik Deteksi (Landmarks)", value=False)

        # Tombol Proses di tengah layout
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Luruskan Wajah (Align Face)", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi wajah dan memproses gambar..."):
                try:
                    # Ubah gambar ke hitam putih untuk algoritma pendeteksi wajah dlib agar lebih cepat
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    
                    # Tahap 1: Deteksi Wajah (Aman dengan Downscaling Otomatis)
                    face = largest_face(detector, gray)
                    if face is None:
                        st.warning("Tidak ada wajah yang terdeteksi pada gambar ini.")
                        st.stop()

                    # Tahap 2: Cari Titik Landmark (68 Titik)
                    points = landmarks_68(predictor, gray, face)
                    
                    # (Opsional) Tampilkan Titik di Gambar Asli
                    if show_landmarks:
                        image_bgr = draw_landmarks(image_bgr, points)
                    
                    # Tahap 3: Cari Titik Tengah Mata Kiri dan Kanan
                    left_eye = eye_center(points, LEFT_EYE)
                    right_eye = eye_center(points, RIGHT_EYE)
                    
                    # Tahap 4: Rotasikan Gambar (serta titik koordinatnya) Berdasarkan Posisi Mata
                    aligned_bgr, _ = rotate_to_align(image_bgr, left_eye, right_eye, points)
                    
                    # Konversi warna gambar hasil ke RGB untuk ditampilkan
                    final_image = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("📏 Gambar Hasil Pemrosesan")
                        st.image(final_image, use_container_width=True)
                        st.success("Wajah berhasil diproses!")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

if __name__ == "__main__":
    main()
