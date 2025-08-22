import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

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
