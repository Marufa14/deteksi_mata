import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import gdown
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Penyakit Mata",
    page_icon="👁️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .result-normal {
        background: linear-gradient(135deg, #1a3a2a, #0d2818);
        border: 1px solid #2ea043;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .result-abnormal {
        background: linear-gradient(135deg, #3a1a1a, #280d0d);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .result-label { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-conf { font-size: 1rem; color: #8b949e; margin: 0; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #c9d1d9;
        margin-bottom: 2px;
    }
    .disclaimer {
        background: #1c2128;
        border-left: 3px solid #d29922;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'model_mata.h5'
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model AI..."):
            url = 'https://drive.google.com/uc?id=1lBNWjnb-1ve61KZJJdWwvtv9mdUdHZp6'
            gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

# ── Grad-CAM function ────────────────────────────────────────
def generate_gradcam(model, img_array, class_idx):
    # Cari layer conv terakhir
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap):
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)

# ── Preprocess ───────────────────────────────────────────────
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img.convert('RGB')) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# ── Show result ──────────────────────────────────────────────
CLASS_NAMES = ['Uveitis', 'Katarak', 'Normal', 'Pterygium']
CLASS_COLORS = {
    'Normal': '#2ea043',
    'Uveitis': '#f85149',
    'Katarak': '#d29922',
    'Pterygium': '#a371f7'
}

def show_result(predictions, image, model):
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = np.max(predictions) * 100

    st.markdown("### Hasil Deteksi")

    if predicted_class == 'Normal':
        st.markdown(f"""
        <div class="result-normal">
            <p class="result-label" style="color:#2ea043">✅ {predicted_class}</p>
            <p class="result-conf">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-abnormal">
            <p class="result-label" style="color:#f85149">⚠️ {predicted_class}</p>
            <p class="result-conf">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Detail Probabilitas:**")
    for i, cls in enumerate(CLASS_NAMES):
        prob = float(predictions[0][i]) * 100
        st.markdown(f"""
        <div class="prob-label">
            <span>{cls}</span><span>{prob:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(predictions[0][i]))

    # Grad-CAM
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Grad-CAM Visualization")
    st.caption("Area berwarna merah/kuning = bagian yang paling diperhatikan model")

    img_array = preprocess_image(image)
    heatmap = generate_gradcam(model, img_array, predicted_idx)
    gradcam_img = overlay_gradcam(image, heatmap)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image.resize((224, 224)), caption="Foto Asli", use_container_width=True)
    with col_b:
        st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Hasil deteksi ini bukan diagnosis medis resmi.
        Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut.
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-title">👁️ Deteksi Penyakit Mata</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem klasifikasi penyakit mata berbasis CNN (MobileNetV2)</p>', unsafe_allow_html=True)

with st.spinner("Memuat model AI..."):
    model = load_model()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  Kamera", "🖼️  Upload Gambar"])

with tab1:
    st.markdown("**Arahkan kamera ke mata, lalu jepret foto.**")
    camera_image = st.camera_input("", label_visibility="collapsed")

    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Foto yang diambil", use_container_width=True)
        with col2:
            with st.spinner("Menganalisis..."):
                img_array = preprocess_image(image)
                predictions = model.predict(img_array, verbose=0)
            show_result(predictions, image, model)

with tab2:
    st.markdown("**Upload gambar mata (JPG/PNG).**")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Gambar diupload", use_container_width=True)
        with col2:
            with st.spinner("Menganalisis..."):
                img_array = preprocess_image(image)
                predictions = model.predict(img_array, verbose=0)
            show_result(predictions, image, model)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#8b949e; font-size:0.8rem;">'
    'Dibuat untuk keperluan Skripsi · Universitas Gunadarma · 2026</p>',
    unsafe_allow_html=True
)