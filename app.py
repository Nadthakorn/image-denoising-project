# ---------------------------------------------------------
# บล็อกที่ 1: นำเข้าไลบรารีและตั้งค่าหน้าเว็บ
# ---------------------------------------------------------
import streamlit as st  # เครื่องมือหลักสำหรับสร้าง Web App
import cv2  # จัดการภาพด้วย OpenCV
import numpy as np  # จัดการอาร์เรย์และตัวเลข
from skimage import util  # เครื่องมือสร้างสัญญาณรบกวน (Noise)
from skimage.metrics import peak_signal_noise_ratio as psnr_func  # ฟังก์ชันคิดคะแนน PSNR
from skimage.metrics import structural_similarity as ssim_func  # ฟังก์ชันคิดคะแนน SSIM
from PIL import Image  # จัดการไฟล์ภาพสำหรับดาวน์โหลด
import io  # แปลงภาพเป็นกระแสข้อมูล (Byte) เพื่อดาวน์โหลด

# ตั้งค่าหน้าเว็บให้แสดงผลแบบเต็มจอ (Wide) และตั้งชื่อแท็บ
st.set_page_config(page_title="Image Restoration Analytics", layout="wide")

# ---------------------------------------------------------
# บล็อกที่ 2: ตกแต่งหน้าตาเว็บด้วย CSS
# ---------------------------------------------------------
# ใช้ HTML/CSS ฝังลงไปเพื่อซ่อนโลโก้และจัดความสวยงาม
st.markdown("""
    <style>
    /* ซ่อนเมนูเริ่มต้นของ Streamlit แต่ยังให้กดเปิด Sidebar ได้ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;} 
    
    /* จัดระยะขอบหน้าจอและเปลี่ยนฟอนต์ */
    .block-container {
        padding-top: 1rem; 
        padding-bottom: 2rem;
        max-width: 90%; 
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* ตกแต่งส่วนหัวข้อใหญ่ */
    .main-header-container {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
    }

    /* ตกแต่งหัวข้อย่อยให้มีขีดสีฟ้าด้านหน้า */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-left: 0.8rem;
        border-left: 4px solid #3B82F6; 
        color: #F8FAFC;
    }
    
    /* ทำมุมรูปภาพให้โค้งมนและมีเงา เวลาเอาเมาส์ชี้ให้ขยับได้ */
    img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    img:hover {
        transform: translateY(-2px); 
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    }
    
    /* จัดตัวหนังสือชื่อวิธีฟิลเตอร์ให้อยู่ตรงกลาง */
    .algo-title {
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1rem;
        color: #E2E8F0;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# บล็อกที่ 3: แถบเมนูด้านข้าง (Sidebar) สำหรับตั้งค่า
# ---------------------------------------------------------
with st.sidebar:
    st.title("Configuration")
    st.markdown("Configure noise parameters and upload source imagery.")
    st.divider() # เส้นคั่น
    
    # ส่วนรับอัปโหลดไฟล์ภาพ
    st.subheader("Source Image")
    uploaded_file = st.file_uploader("Upload an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    # ส่วนเลือกตั้งค่า Noise
    st.subheader("Noise Simulation")
    noise_type = st.selectbox(
        "Noise Distribution", 
        ["Gaussian", "Salt & Pepper", "Speckle", "Poisson"]
    )
    
    # สไลเดอร์ปรับระดับความรุนแรงของ Noise
    noise_level = st.slider(
        "Noise Variance / Amount", 
        min_value=0.0, 
        max_value=0.2, 
        value=0.05,
        step=0.01
    )

# ---------------------------------------------------------
# บล็อกที่ 4: ส่วนหัวของหน้าเว็บหลัก
# ---------------------------------------------------------
st.markdown("""
    <div class="main-header-container">
        <div class="main-title">Image Restoration & Denoising Analytics</div>
        <div class="main-subtitle">Comparative analysis of spatial filtering algorithms using <b>PSNR</b> and <b>SSIM</b> metrics.</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# บล็อกที่ 5: ระบบประมวลผล (ทำงานต่อเมื่อมีรูปถูกอัปโหลด)
# ---------------------------------------------------------
if uploaded_file is not None:
    # อ่านไฟล์ภาพที่ผู้ใช้อัปโหลดมาเป็นไบต์ แล้วแปลงเป็นอาเรย์สี RGB
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_clean = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
    
    # ปรับสเกลภาพเป็น 0-1 สำหรับส่งให้ฟังก์ชันใส่ Noise
    img_norm = img_clean.astype(float) / 255.0
    
    # เช็คว่าผู้ใช้เลือก Noise แบบไหน แล้วจำลองแบบนั้นใส่ภาพ
    if noise_type == "Gaussian":
        noisy = util.random_noise(img_norm, mode='gaussian', var=noise_level)
    elif noise_type == "Salt & Pepper":
        noisy = util.random_noise(img_norm, mode='s&p', amount=noise_level)
    elif noise_type == "Speckle":
        noisy = util.random_noise(img_norm, mode='speckle', var=noise_level)
    else:
        noisy = util.random_noise(img_norm, mode='poisson')
    
    # แปลงภาพที่ติด Noise กลับเป็นสเกล 0-255
    img_noisy = (noisy * 255).astype(np.uint8)
    
    # คำนวณคะแนนพื้นฐาน (ภาพต้นฉบับ vs ภาพติด Noise)
    base_psnr = psnr_func(img_clean, img_noisy)
    # ใช้ try-except กัน error เผื่อรูปเล็กเกินไปสำหรับ SSIM
    try:
        base_ssim = ssim_func(img_clean, img_noisy, channel_axis=2, data_range=255)
    except Exception:
        base_ssim = ssim_func(img_clean, img_noisy, channel_axis=2, win_size=3, data_range=255)

    # ---------------------------------------------------------
    # บล็อกที่ 6: แสดงภาพต้นฉบับเปรียบเทียบกับภาพติด Noise
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>1. Baseline Assessment</div>", unsafe_allow_html=True)
    
    # แบ่งพื้นที่เป็น 3 คอลัมน์ (ภาพเดิม, ภาพ Noise, กล่องคะแนน)
    col_img1, col_img2, col_metrics = st.columns([1.5, 1.5, 1])
    
    with col_img1:
        st.image(img_clean, caption="Original Signal (Ground Truth)", use_container_width=True)
        
    with col_img2:
        st.image(img_noisy, caption=f"Degraded Signal ({noise_type})", use_container_width=True)
        
    with col_metrics:
        # แสดงกล่องคะแนน PSNR และ SSIM ของภาพที่ติด Noise
        with st.container(border=True):
            st.markdown("<h5 style='color: #E2E8F0; margin-bottom: 1rem;'>Baseline Metrics</h5>", unsafe_allow_html=True)
            st.metric(label="PSNR (Higher is better)", value=f"{base_psnr:.2f} dB")
            st.divider()
            st.metric(label="SSIM (Closer to 1 is better)", value=f"{base_ssim:.4f}")

    # ---------------------------------------------------------
    # บล็อกที่ 7: ตัวกรองภาพและเปรียบเทียบคะแนน
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>2. Algorithm Performance Comparison</div>", unsafe_allow_html=True)
    
    # โชว์วงล้อโหลดข้อมูลระหว่างที่ระบบกำลังคิดเลข
    with st.spinner('Processing spatial filters and computing structural similarities...'):
        
        # จัดเตรียมฟิลเตอร์ทั้ง 5 วิธีด้วย OpenCV
        filters = {
            "Mean Filter": cv2.blur(img_noisy, (5, 5)),
            "Gaussian Filter": cv2.GaussianBlur(img_noisy, (5, 5), 0),
            "Median Filter": cv2.medianBlur(img_noisy, 5),
            "Bilateral Filter": cv2.bilateralFilter(img_noisy, 9, 75, 75),
            "NL-Means (Fast)": cv2.fastNlMeansDenoisingColored(img_noisy, None, 10, 10, 7, 21)
        }

        filter_results = {}
        best_filter_name = ""
        max_psnr = 0
        best_img = None

        # วนลูปคิดคะแนน PSNR และ SSIM ให้ทุกฟิลเตอร์
        for name, f_img in filters.items():
            current_psnr = psnr_func(img_clean, f_img)
            try:
                current_ssim = ssim_func(img_clean, f_img, channel_axis=2, data_range=255)
            except Exception:
                current_ssim = ssim_func(img_clean, f_img, channel_axis=2, win_size=3, data_range=255)
            
            # เก็บข้อมูลคะแนนและส่วนต่าง (Delta) ไว้แสดงผล
            filter_results[name] = {
                "img": f_img,
                "psnr": current_psnr,
                "ssim": current_ssim,
                "psnr_delta": current_psnr - base_psnr,
                "ssim_delta": current_ssim - base_ssim
            }

            # หาว่าวิธีไหนได้คะแนน PSNR สูงสุด
            if current_psnr > max_psnr:
                max_psnr = current_psnr
                best_filter_name = name
                best_img = f_img

        # แบ่งเป็น 5 คอลัมน์สำหรับโชว์ 5 วิธี
        f_cols = st.columns(5)
        for i, (name, data) in enumerate(filter_results.items()):
            with f_cols[i]:
                # โชว์ชื่อวิธีและรูปภาพ
                st.markdown(f"<div class='algo-title'>{name}</div>", unsafe_allow_html=True)
                st.image(data["img"], use_container_width=True)
                
                # ถ้าวิธีนี้ได้คะแนนสูงสุด ให้เติมสัญลักษณ์ไฟ 🔥
                if name == best_filter_name:
                    display_psnr = f"{data['psnr']:.2f} 🔥"
                else:
                    display_psnr = f"{data['psnr']:.2f}"
                
                # โชว์คะแนนพร้อมตัวเปรียบเทียบว่าดีขึ้นเท่าไหร่
                st.metric(label="PSNR (dB)", value=display_psnr, delta=f"{data['psnr_delta']:.2f}")
                st.metric(label="SSIM", value=f"{data['ssim']:.4f}", delta=f"{data['ssim_delta']:.4f}")

    # ---------------------------------------------------------
    # บล็อกที่ 8: สรุปผลลัพธ์และปุ่มดาวน์โหลด
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>3. Optimal Restoration Result</div>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([3, 1])
    
    with res_col1:
        # โชว์กล่องข้อความสีเขียวสรุปวิธีที่ดีที่สุด
        st.success(f"Algorithm with highest peak signal-to-noise ratio: **{best_filter_name}** ({max_psnr:.2f} dB)")
    
    with res_col2:
        # เตรียมแปลงรูปที่ดีที่สุดกลับเป็นไฟล์ .png
        img_pil = Image.fromarray(best_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # สร้างปุ่มสำหรับกดดาวน์โหลดภาพไปที่เครื่อง
        st.download_button(
            label="Download Restored Image",
            data=byte_im,
            file_name=f"restored_{best_filter_name.replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True
        )

# ---------------------------------------------------------
# บล็อกที่ 9: หน้าจอ Empty State (โชว์ตอนที่ยังไม่มีรูปอัปโหลด)
# ---------------------------------------------------------
else:
    st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 50vh;
            text-align: center;
            color: #94A3B8;
        ">
            <div style="
                padding: 2.5rem 4rem;
                background-color: rgba(30, 41, 59, 0.4);
                border: 2px dashed rgba(71, 85, 105, 0.5);
                border-radius: 12px;
            ">
                <span style="font-size: 1.3rem; font-weight: 600; color: #E2E8F0; letter-spacing: 0.5px;">System Ready</span><br>
                <span style="font-size: 1rem; margin-top: 0.8rem; display: inline-block;">Please upload a source image via the configuration panel to initiate analysis.</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
