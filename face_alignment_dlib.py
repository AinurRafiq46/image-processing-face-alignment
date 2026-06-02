import math
from pathlib import Path
import cv2
import dlib
import numpy as np

# ==========================================
# 1. KONSTANTA & KONFIGURASI AWAL
# ==========================================
LEFT_EYE = range(36, 42)
RIGHT_EYE = range(42, 48)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_project_root() -> Path:
    return Path(__file__).resolve().parent

# ==========================================
# 2. FUNGSI PENDUKUNG FILE & MODEL (SETUP)
# ==========================================
def find_predictor_file() -> Path:
    root = get_project_root()
    candidates = [
        root / "models" / "shape_predictor_68_face_landmarks.dat",
        root / "shape_predictor_68_face_landmarks.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "File shape_predictor_68_face_landmarks.dat tidak ditemukan. "
        "Simpan file itu di folder models atau di folder project."
    )

def load_models() -> tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(find_predictor_file()))
    return detector, predictor

def get_input_images() -> list[Path]:
    image_dir = get_project_root() / "image"
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {image_dir}")
    
    images = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"Tidak ada gambar valid di folder: {image_dir}")
    return images

# ==========================================
# 3. FUNGSI INTI PEMROSESAN CITRA (IMAGE PROCESSING)
# ==========================================
def largest_face(detector: dlib.fhog_object_detector, gray: np.ndarray) -> dlib.rectangle:
    faces = detector(gray, 1)
    if not faces:
        raise ValueError("Tidak ada wajah terdeteksi.")
    # Mengambil wajah terbesar jika ada beberapa wajah dalam satu frame gambar
    return max(faces, key=lambda face: face.width() * face.height())

def landmarks_68(predictor: dlib.shape_predictor, gray: np.ndarray, face: dlib.rectangle) -> np.ndarray:
    shape = predictor(gray, face)
    points = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        points[i] = (shape.part(i).x, shape.part(i).y)
    return points

def eye_center(points: np.ndarray, eye_indexes: range) -> np.ndarray:
    # Menghitung rata-rata nilai koordinat mata (Centroid) untuk hasil sudut yang lebih stabil
    return points[list(eye_indexes)].mean(axis=0)

def rotate_to_align(image: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
    # Perhitungan delta X dan delta Y antara mata kanan dan kiri
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Menghitung derajat kemiringan sudut
    angle = math.degrees(math.atan2(dy, dx))
    center = tuple(((left_eye + right_eye) / 2).tolist())

    # Membuat matriks rotasi dan mengeksekusi Affine Transformation (WarpAffine)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE, # Duplikasi piksel tepi agar tidak ada area kosong hitam yang tajam
    )

# ==========================================
# 4. FUNGSI VISUALISASI & TAMPILAN (DISPLAY)
# ==========================================
def add_label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.putText(labeled, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return labeled

def resize_for_display(image: np.ndarray, width: int = 280) -> np.ndarray:
    h, w = image.shape[:2]
    new_height = int((width / w) * h)
    return cv2.resize(image, (width, new_height))

def make_pair(original: np.ndarray, aligned: np.ndarray, title: str) -> np.ndarray:
    # Mengubah ukuran gambar asli dan yang lurus agar rapi saat disandingkan side-by-side
    original_view = add_label(resize_for_display(original), f"Asli - {title}")
    aligned_view = add_label(resize_for_display(aligned), f"Lurus - {title}")

    max_height = max(original_view.shape[0], aligned_view.shape[0])

    def pad_bottom(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == max_height:
            return image
        pad = max_height - image.shape[0]
        return cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Gabung gambar secara horizontal (kiri: Asli, kanan: Lurus)
    return np.hstack([pad_bottom(original_view), pad_bottom(aligned_view)])

# ==========================================
# 5. ALUR UTAMA PROGRAM (MAIN FUNCTION)
# ==========================================
def main() -> None:
    detector, predictor = load_models()
    image_paths = get_input_images()

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] Gagal membaca: {image_path.name}")
            continue

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = largest_face(detector, gray)
            points = landmarks_68(predictor, gray, face)

            # Proses inti mencari koordinat mata dan merotasikan gambar
            left_eye = eye_center(points, LEFT_EYE)
            right_eye = eye_center(points, RIGHT_EYE)
            aligned = rotate_to_align(image, left_eye, right_eye)

            # Menumpuk pasang gambar ke dalam daftar hasil
            rows.append(make_pair(image, aligned, image_path.name))
            print(f"[OK] Diproses: {image_path.name}")
        except Exception as error:
            print(f"[SKIP] {image_path.name}: {error}")

    if not rows:
        raise RuntimeError("Tidak ada gambar yang berhasil diproses.")

    # Menyatukan semua pasang gambar secara vertikal menjadi satu sheet laporan besar
    sheet = np.vstack(rows)
    cv2.imshow("Perbandingan Asli vs Lurus", sheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()