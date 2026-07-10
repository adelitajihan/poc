import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# ==========================
# FIREBASE
# ==========================
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))

    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": "https://pendeteksi-daun-default-rtdb.asia-southeast1.firebasedatabase.app/"
        },
    )

database = db.reference("hasil_scan")

# ==========================
# STREAMLIT
# ==========================
st.set_page_config(
    page_title="Scanner Penyakit Daun",
    layout="wide"
)

st.title("🌿 Scanner Penyakit Daun")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")
    # Kalau modelmu ada di folder yang sama dengan app.py gunakan:
    # return YOLO("best.pt")

model = load_model()

# ==========================
# PILIH GAMBAR
# ==========================
mode = st.radio(
    "Pilih Sumber Gambar",
    ["📷 Kamera", "🖼 Upload"]
)

img_file = None

if mode == "📷 Kamera":
    img_file = st.camera_input("Ambil Foto Daun")
else:
    img_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

# ==========================
# DETEKSI
# ==========================
if img_file is not None:

    image = Image.open(img_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("Sedang mendeteksi..."):
        results = model(img_array)

    st.image(
        results[0].plot(),
        caption="Hasil Deteksi",
        use_container_width=True
    )

    if len(results[0].boxes) == 0:

        st.warning("Tidak ada daun yang terdeteksi.")

    else:

        names = results[0].names

        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            disease = names[cls]

            status = "Sehat" if disease.lower() == "healthy" else "Sakit"

            st.success(f"{disease} ({conf:.2%})")

            data = {
                "penyakit": disease,
                "confidence": round(conf, 4),
                "status": status,
                "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            database.push(data)

# ==========================
# RIWAYAT
# ==========================
st.divider()
st.subheader("📋 Riwayat Scan")

history = database.get()

if history:

    history = list(history.items())
    history.reverse()

    for key, value in history:

        with st.expander(
            f"{value['waktu']} - {value['penyakit']}"
        ):
            st.write("**Status :**", value["status"])

else:

    st.info("Belum ada data.")
