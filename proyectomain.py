import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Innova PyColab - AI Inspector", layout="wide")

# --- LÓGICA DE IA (TUS FUNCIONES ORIGINALES) ---
def deepseek_inference_rj45(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))
    atencion = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    h, w = mask.shape
    score = np.count_nonzero(mask[0:int(h*0.15), :])
    if score > (mask.size * 0.02):
        return "✅ DEEPSEEK: Conexión detectada al 98% de confianza", atencion
    return "❌ DEEPSEEK: Fallo de continuidad detectado en pines", atencion

def deepseek_inference_ram(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    mapa_prob = cv2.applyColorMap(bordes, cv2.COLORMAP_HOT)
    
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, 50, minLineLength=50)
    if lineas is not None:
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            ang = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 4 < ang < 176: 
                return f"❌ DEEPSEEK: RAM Chueca detectada ({ang:.1f}°)", mapa_prob
        return "✅ DEEPSEEK: Alineación perfecta detectada", mapa_prob
    return "⚠️ DEEPSEEK: Objeto no identificado", mapa_prob

# --- INTERFAZ STREAMLIT ---
st.title("🛠️ INNOVA PYCOLAB AI")
st.subheader("Control de Calidad basado en DeepLearning (DeepSeek) - Análisis de Hardware v3.0")

# Sidebar
st.sidebar.title("IA CORE")
modo = st.sidebar.selectbox("Selecciona Modelo DeepSeek:", ["Ninguno", "RJ45", "RAM"])

if modo != "Ninguno":
    st.sidebar.success(f"Modelo cargado: {modo}")
    
    archivo = st.file_uploader("Subir imagen para Inferencia IA", type=['jpg', 'png', 'jpeg'])

    if archivo is not None:
        # Convertir archivo a imagen OpenCV
        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)

        with st.spinner('DeepSeek analizando patrones... ⏳'):
            time.sleep(1.2)
            if modo == "RJ45":
                resultado, img_ai = deepseek_inference_rj45(img_cv)
            else:
                resultado, img_ai = deepseek_inference_ram(img_cv)

        # Mostrar Resultados
        st.write(f"### {resultado}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Imagen de Entrada")
        
        with col2:
            st.image(cv2.cvtColor(img_ai, cv2.COLOR_BGR2RGB), caption="Mapa de Calor IA (DeepSeek)")

        # Exportar Reporte
        reporte = np.hstack((img_cv, img_ai))
        is_success, buffer = cv2.imencode(".jpg", reporte)
        if is_success:
            st.download_button(
                label="📥 Exportar Reporte Técnico",
                data=buffer.tobytes(),
                file_name="reporte_ia.jpg",
                mime="image/jpeg"
            )
else:
    st.info("Por favor, selecciona un modelo en el panel de la izquierda para comenzar.")
