import streamlit as st
import cv2
import numpy as np
from skimage import util
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from PIL import Image

# ปรับแต่งหน้าเว็บ
st.set_page_config(page_title="Image Denoising Lab", layout="wide")

st.title("🛡️ Ultimate Image Denoising Lab 2026")
st.write("อัปโหลดภาพเพื่อทดสอบการกำจัด Noise และวัดผลด้วยค่า PSNR")

# แถบเมนูด้านข้าง
st.sidebar.header("⚙️ ตั้งค่า")
uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])
noise_type = st.sidebar.selectbox("เลือกประเภท Noise", ["Gaussian", "Salt & Pepper", "Speckle", "Poisson"])
noise_level = st.sidebar.slider("ความแรงของ Noise", 0.0, 0.2, 0.05)

if uploaded_file is not None:
    # จัดการรูปภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_clean = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
    
    # ใส่ Noise
    img_norm = img_clean.astype(float) / 255.0
    if noise_type == "Gaussian":
        noisy = util.random_noise(img_norm, mode='gaussian', var=noise_level)
    elif noise_type == "Salt & Pepper":
        noisy = util.random_noise(img_norm, mode='s&p', amount=noise_level)
    elif noise_type == "Speckle":
        noisy = util.random_noise(img_norm, mode='speckle', var=noise_level)
    else:
        noisy = util.random_noise(img_norm, mode='poisson')
    
    img_noisy = (noisy * 255).astype(np.uint8)
    n_psnr = psnr_func(img_clean, img_noisy)

    # แสดงผลภาพเปรียบเทียบ
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ ต้นฉบับ")
        st.image(img_clean, use_container_width=True)
    with col2:
        st.subheader(f"❌ Noise: {noise_type}")
        st.image(img_noisy, use_container_width=True)
        st.error(f"Input PSNR: {n_psnr:.2f} dB")

    st.divider()
    
    # กรองภาพด้วย 5 ฟิลเตอร์
    st.header("🧪 เปรียบเทียบ 5 ฟิลเตอร์")
    filters = {
        "Mean": cv2.blur(img_noisy, (5, 5)),
        "Gaussian": cv2.GaussianBlur(img_noisy, (5, 5), 0),
        "Median": cv2.medianBlur(img_noisy, 5),
        "Bilateral": cv2.bilateralFilter(img_noisy, 9, 75, 75),
        "NL-Means": cv2.fastNlMeansDenoisingColored(img_noisy, None, 10, 10, 7, 21)
    }

    cols = st.columns(5)
    best_f, max_p = "", 0

    for i, (name, f_img) in enumerate(filters.items()):
        f_psnr = psnr_func(img_clean, f_img)
        with cols[i]:
            st.image(f_img, caption=f"{name}", use_container_width=True)
            st.metric("PSNR", f"{f_psnr:.2f}", delta=f"{f_psnr-n_psnr:.2f}")
            if f_psnr > max_p:
                max_p = f_psnr
                best_f = name

    st.success(f"🏆 ฟิลเตอร์ที่เจ๋งที่สุดสำหรับรูปนี้คือ: **{best_f}**")
else:
    st.info("👈 เริ่มต้นโดยการอัปโหลดรูปภาพที่เมนูด้านซ้ายได้เลย!")
