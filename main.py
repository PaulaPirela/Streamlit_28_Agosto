import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import pytesseract

def analizar_imagen(imagen_pil):
    """
    Analiza una imagen cargada por el usuario para extraer texto y colores predominantes.

    Args:
        imagen_pil (PIL.Image.Image): La imagen cargada por el usuario.

    Returns:
        tuple: Una tupla que contiene el texto extraído y la imagen con los colores predominantes.
    """
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    imagen_rgb = np.array(imagen_pil)

    # --- Extracción de texto con OCR ---
    st.subheader("Extracción de texto de la imagen:")
    try:
        texto_extraido = pytesseract.image_to_string(imagen_pil)
        if not texto_extraido.strip():
            st.info("No se encontró texto en la imagen.")
            texto_extraido = "No se detectó texto."
        else:
            st.text_area("Texto extraído:", texto_extraido, height=200)
    except pytesseract.TesseractNotFoundError:
        st.error("Error: Tesseract no está instalado o no se encuentra en la ruta del sistema.")
        st.info("Por favor, asegúrate de que Tesseract esté instalado y configurado correctamente.")
        return "Error de Tesseract", imagen_cv
    except Exception as e:
        st.error(f"Error durante el OCR: {e}")
        return "Error al procesar el texto.", imagen_cv

    # --- Análisis de colores predominantes ---
    st.subheader("Análisis de colores predominantes:")
    try:
        # Redimensionar la imagen para un procesamiento más rápido
        pixeles = imagen_rgb.reshape((-1, 3))

        # Usar K-Means para encontrar los 5 colores más comunes
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixeles)
        colores_predominantes = kmeans.cluster_centers_.astype(int)

        st.write("Colores predominantes (RGB):")
        for color in colores_predominantes:
            st.markdown(f"<span style='background-color:rgb({color [0]},{color [1]},{color [2]}); color:white;'>rgb({color [0]}, {color [1]}, {color [2]})</span>", unsafe_allow_html=True)

        # --- Visualización de los colores ---
        altura_barra = 50
        ancho_paleta = 300
        paleta = np.zeros((altura_barra, ancho_paleta, 3), np.uint8)
        paso = ancho_paleta // len(colores_predominantes)

        for i, color in enumerate(colores_predominantes):
            paleta[:, i * paso:(i + 1) * paso, :] = color

        # Convertir la paleta a un formato que se pueda mostrar con la imagen original
        paleta_rgb = cv2.cvtColor(paleta, cv2.COLOR_RGB2BGR)
        paleta_pil = Image.fromarray(paleta_rgb)
        st.image(paleta_pil, caption="Paleta de colores predominantes")

        # Mostrar la imagen original (opcional)
        st.subheader("Imagen cargada:")
        st.image(imagen_pil, caption="Imagen original", use_column_width=True)

        return texto_extraido, cv2.cvtColor(np.vstack((imagen_cv, paleta_rgb)), cv2.COLOR_BGR2RGB)

    except Exception as e:
        st.error(f"Error durante el análisis de color: {e}")
        st.image(imagen_pil, caption="Imagen original", use_column_width=True)
        return texto_extraido, imagen_cv

if __name__ == "__main__":
    st.title("Asistente Inteligente de Imágenes con OCR")

    uploaded_file = st.file_uploader("Carga una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            imagen_pil = Image.open(uploaded_file)
            texto, imagen_analizada_cv = analizar_imagen(imagen_pil)

            # Opción para descargar la imagen analizada
            imagen_analizada_pil = Image.fromarray(imagen_analizada_cv)
            buffered = BytesIO()
            imagen_analizada_pil.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="imagen_analizada.png">Descargar imagen analizada</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Opción para descargar el texto extraído
            b64_texto = base64.b64encode(texto.encode()).decode()
            href_texto = f'<a href="data:file/txt;base64,{b64_texto}" download="texto_extraido.txt">Descargar texto extraído</a>'
            st.markdown(href_texto, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")

from io import BytesIO
import base64
