import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
from datetime import datetime
import pymongo
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Clasificador Fauna Huánuco", page_icon="🐾", layout="wide")

CLASES = ["Cebus yuracus", "Mono choro de cola amarilla", "Oso de anteojos"]
NOMBRE_COMUN = ["Capuchino de frente blanca", "Mono choro de cola amarilla", "Oso de anteojos / Oso andino"]
EMOJIS = ["🐒", "🐵", "🐻"]
COLORES = ["#D9A441", "#4E7B4E", "#8B5E3C"]

# ---------- ESTILOS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0F1F14 0%, #1A2E1F 100%);
    color: #EDEDED;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #F2E9DC;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #B9C9B2;
    font-size: 1.05rem;
}
.badge {
    display: inline-block;
    background: #D9A441;
    color: #1A2E1F;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

.species-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    transition: transform 0.2s;
}
.species-card:hover { transform: translateY(-4px); }
.species-card .emoji { font-size: 2.2rem; }
.species-card .nombre-cientifico { font-style: italic; color: #D9A441; font-size: 0.85rem; margin-top: 0.3rem; }
.species-card .nombre-comun { font-weight: 600; color: #F2E9DC; font-size: 0.95rem; }

.result-card {
    background: linear-gradient(135deg, rgba(78,123,78,0.25), rgba(217,164,65,0.15));
    border: 1px solid rgba(217,164,65,0.4);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}
.result-card h2 {
    color: #F2E9DC;
    margin-bottom: 0.2rem;
}
.result-card .confianza {
    font-size: 2.2rem;
    font-weight: 700;
    color: #D9A441;
}

[data-testid="stSidebar"] {
    background: #142018;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 1rem;
}

[data-testid="stSidebar"] * {
    color: #EDEDED !important;
}
[data-testid="stSidebar"] h3 {
    color: #F2E9DC !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}
[data-testid="stCaptionContainer"] {
    color: #9DB09A !important;
}

/* --- Quitar espacio blanco superior (header nativo de Streamlit) --- */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem !important;
}
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}
.block-container {
    padding-top: 2rem !important;
}

/* --- Zona de carga estilo "IA" --- */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(20,38,28,0.9), rgba(8,18,12,0.95)) !important;
    border: 2px dashed #D4AF37 !important;
    border-radius: 18px !important;
    padding: 1.6rem !important;
    animation: pulseGlowIA 3s ease-in-out infinite;
}
@keyframes pulseGlowIA {
    0%   { box-shadow: 0 0 0px rgba(212,175,55,0.25); }
    50%  { box-shadow: 0 0 22px rgba(212,175,55,0.45); }
    100% { box-shadow: 0 0 0px rgba(212,175,55,0.25); }
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] * {
    color: #EDEDED !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #1F6B3A, #0B2E1A) !important;
    color: #F2E9DC !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #4ADE80 !important;
    color: #FFFFFF !important;
}

.section-subtitle {
    text-align: center;
    color: #F2E9DC;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.8rem 0 1.2rem 0;
}
.step-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.step-card:hover {
    transform: translateY(-4px);
    border-color: rgba(212,175,55,0.5);
}
.step-card .step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.2rem;
    height: 2.2rem;
    background: linear-gradient(135deg, #D9A441, #B9842E);
    color: #1A2E1F;
    font-weight: 700;
    border-radius: 50%;
    margin-bottom: 0.6rem;
    font-size: 1rem;
}
.step-card .step-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
.step-card h4 {
    color: #F2E9DC;
    margin: 0.3rem 0;
    font-size: 1.05rem;
}
.step-card p {
    color: #B9C9B2;
    font-size: 0.88rem;
    margin: 0;
}

[data-testid="stImage"] img {
    max-height: 420px;
    width: auto;
    object-fit: contain;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

[data-testid="stCameraInput"] {
    max-width: 380px;
    margin: 0 auto;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    max-height: 380px;
    object-fit: contain;
    border-radius: 14px;
}

[data-testid="stCameraInput"] button {
}
[data-testid="stCameraInput"] button:hover {
}
[data-testid="stCameraInput"] * {
}
[data-testid="stRadio"] * {
}
[data-testid="stWidgetLabel"] p {
}
[data-testid="stCameraInput"] label p {
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    return load_model("mejor_modelo.keras")

modelo = cargar_modelo()

@st.cache_resource
def conectar_mongo():
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        st.sidebar.error("⚠️ MONGO_URI no está definida")
        return None
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client["fauna_huanuco"]
        return db["predicciones"]
    except Exception as e:
        st.sidebar.error(f"⚠️ Error Mongo: {e}")
        return None

coleccion = conectar_mongo()

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_image()
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "PRUEBA - sin IA todavia", fill=(0, 255, 0))
        return av.VideoFrame.from_image(img)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📋 Ficha técnica")
    st.markdown("""
    **Arquitectura:** MobileNetV2 + Transfer Learning
    **Épocas de entrenamiento:** 100
    **Entorno:** Google Colab (GPU T4)
    **Clases:** 3
    **Mejor val_accuracy:** 91.3%
    """)
    st.divider()
    st.markdown("### 🌎 Contexto")
    st.markdown("""
    Proyecto académico sobre mamíferos en peligro
    de la región **Huánuco, Perú**.

    Inspirado en la Línea 3 (Conservación de la
    biodiversidad y protección de fauna silvestre)
    del concurso **INNOBEST Huánuco 2026**.
    """)
    st.divider()
    if coleccion is not None:
        try:
            total = coleccion.count_documents({})
            st.markdown("### 📊 Base de datos")
            st.markdown(f"**{total}** predicciones registradas en MongoDB")
        except Exception:
            pass
    st.divider()
    st.caption("Desarrollado por Aldahir Pablo Silva Villar · SENATI Huánuco")

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="badge">🌿 CONSERVACIÓN · HUÁNUCO 2026</div>
    <h1>🐾 Clasificador de Fauna en Peligro</h1>
    <p>Identifica mamíferos amenazados de la región Huánuco a partir de una foto, usando un modelo de visión por computadora entrenado con GPU.</p>
</div>
""", unsafe_allow_html=True)

# ---------- TARJETAS DE ESPECIES ----------
cols = st.columns(3)
for col, emoji, cientifico, comun in zip(cols, EMOJIS, CLASES, NOMBRE_COMUN):
    with col:
        st.markdown(f"""
        <div class="species-card">
            <div class="emoji">{emoji}</div>
            <div class="nombre-comun">{comun}</div>
            <div class="nombre-cientifico">{cientifico}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()

# ---------- CARGA DE IMAGEN ----------
st.markdown("#### ✨ Sube una foto o usa la camara")
metodo = st.radio("Metodo de entrada", ["Subir foto", "Usar camara", "Camara en vivo (prueba)"], horizontal=True, label_visibility="collapsed")

archivo = None

if metodo == "Subir foto":
    archivo = st.file_uploader("Formatos aceptados: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
elif metodo == "Usar camara":
    archivo = st.camera_input("Toma una foto con tu camara")
else:
    st.info("Prueba de video en vivo, sin IA todavia")
    webrtc_streamer(
        key="camara-viva",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

if archivo is not None:
    col_img, col_result = st.columns([1, 1.3])

    imagen = Image.open(archivo).convert("RGB")
    with col_img:
        st.image(imagen, width=380, caption="Imagen cargada")

    img_resized = imagen.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analizando con el modelo..."):
        prediccion = modelo.predict(img_array)

    indice = np.argmax(prediccion)
    clase_predicha = NOMBRE_COMUN[indice]
    confianza = float(np.max(prediccion) * 100)

    UMBRAL_CONFIANZA = 60.0

    with col_result:
        if confianza < UMBRAL_CONFIANZA:
            st.markdown(f"""
            <div class="result-card">
                <h2>❓ Especie no identificada</h2>
                <div class="confianza">{confianza:.1f}%</div>
                <p style="color:#B9C9B2;">Confianza insuficiente: la imagen podria no corresponder a ninguna de las 3 especies registradas.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h2>{EMOJIS[indice]} {clase_predicha}</h2>
                <div class="confianza">{confianza:.1f}%</div>
                <p style="color:#B9C9B2;">de confianza</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Probabilidad por especie")
    for i, (comun, color) in enumerate(zip(NOMBRE_COMUN, COLORES)):
        prob = float(prediccion[0][i] * 100)
        st.markdown(f"**{EMOJIS[i]} {comun}** — {prob:.2f}%")
        st.progress(prob / 100)

    if coleccion is not None and confianza >= UMBRAL_CONFIANZA:
        try:
            coleccion.insert_one({
                "especie_cientifica": CLASES[indice],
                "especie_comun": clase_predicha,
                "confianza": confianza,
                "fecha": datetime.utcnow()
            })
        except Exception:
            pass
else:
    st.markdown('<div class="section-subtitle">¿Cómo funciona?</div>', unsafe_allow_html=True)
    pasos = [
        ("1", "📤", "Sube tu foto", "Elige una imagen clara del mamífero que quieres identificar, en JPG o PNG."),
        ("2", "🧠", "La IA analiza", "El modelo MobileNetV2, entrenado con GPU en Google Colab, procesa la imagen en segundos."),
        ("3", "✅", "Obtén el resultado", "Verás la especie detectada junto al porcentaje de confianza del modelo."),
    ]
    paso_cols = st.columns(3)
    for col, (numero, icono, titulo, descripcion) in zip(paso_cols, pasos):
        with col:
            st.markdown(f'''
            <div class="step-card">
                <div class="step-number">{numero}</div>
                <div class="step-icon">{icono}</div>
                <h4>{titulo}</h4>
                <p>{descripcion}</p>
            </div>
            ''', unsafe_allow_html=True)
    st.write("")

st.write("")
st.divider()
st.caption("Proyecto académico · Clasificación de fauna silvestre en peligro de Huánuco, Perú · Modelo entrenado en Google Colab con GPU, 100 épocas, transfer learning (MobileNetV2)")
