import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from io import BytesIO      # <-- LÍNEA AÑADIDA
import base64           # <-- LÍNEA AÑADIDA

def analizar_imagen(imagen_pil):
    """
    Analiza una imagen cargada por el usuario para extraer texto y colores predominantes.

    Args:
        imagen_pil (PIL.Image.Image): La imagen cargada por el usuario.

    Returns:
        tuple: Una tupla que contiene el texto extraído y la imagen con los colores predominantes.
    """
    # Convertir la imagen de PIL a formatos de OpenCV para procesamiento
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    imagen_rgb = np.array(imagen_pil)

    # --- Extracción de texto con OCR ---
    st.subheader("1. Texto extraído con OCR")
    try:
        # Es posible que necesites configurar la ruta del ejecutable de Tesseract si no está en el PATH
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Ejemplo para Windows
        texto_extraido = pytesseract.image_to_string(imagen_pil, lang='spa+eng') # Intenta español e inglés
        if not texto_extraido.strip():
            st.info("No se encontró texto en la imagen.")
            texto_extraido = "No se detectó texto."
        else:
            st.text_area("Texto encontrado:", texto_extraido, height=200)
    except pytesseract.TesseractNotFoundError:
        st.error("Error: Tesseract no está instalado o no se encuentra en la ruta del sistema.")
        st.info("Por favor, instala Tesseract y/o configura la ruta del ejecutable en el script.")
        return "Error de Tesseract", imagen_cv
    except Exception as e:
        st.error(f"Ocurrió un error durante el OCR: {e}")
        return "Error al procesar el texto.", imagen_cv

    # --- Análisis de colores predominantes ---
    st.subheader("2. Colores Predominantes")
    try:
        # Redimensionar la imagen para que el clustering sea más rápido
        pixeles = imagen_rgb.reshape((-1, 3))

        # Usar K-Means para encontrar los 5 colores más comunes
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixeles)
        colores_predominantes = kmeans.cluster_centers_.astype(int)

        # Crear una paleta de colores para mostrar
        altura_barra = 50
        ancho_paleta = 300
        paleta = np.zeros((altura_barra, ancho_paleta, 3), np.uint8)
        paso = ancho_paleta // len(colores_predominantes)

        cols = st.columns(len(colores_predominantes))
        for i, color in enumerate(colores_predominantes):
            with cols[i]:
                # Muestra un pequeño cuadrado de color y su código RGB
                st.markdown(f"<div style='background-color:rgb({color[0]},{color[1]},{color[2]}); width:30px; height:30px; border: 1px solid black;'></div>", unsafe_allow_html=True)
                st.write(f"RGB({color[0]}, {color[1]}, {color[2]})")
            
            # Dibuja el color en la imagen de la paleta
            paleta[:, i * paso:(i + 1) * paso, :] = color

        # Prepara la imagen combinada para el output
        # Redimensionar imagen original para que coincida con el ancho de la paleta
        escala = ancho_paleta / imagen_cv.shape[1]
        nueva_altura = int(imagen_cv.shape[0] * escala)
        imagen_redimensionada = cv2.resize(imagen_cv, (ancho_paleta, nueva_altura))
        paleta_bgr = cv2.cvtColor(paleta, cv2.COLOR_RGB2BGR)
        imagen_final_cv = np.vstack((imagen_redimensionada, paleta_bgr))

    except Exception as e:
        st.error(f"Ocurrió un error durante el análisis de color: {e}")
        # Si falla el análisis de color, devuelve la imagen original
        imagen_final_cv = imagen_cv

    return texto_extraido, imagen_final_cv

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Asistente Inteligente de Imágenes 🖼️")
    st.write("Carga una imagen para extraer su texto y analizar sus colores predominantes.")

    uploaded_file = st.file_uploader("Selecciona un archivo de imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            imagen_pil = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagen Original")
                st.image(imagen_pil, caption="Imagen cargada por el usuario.", use_column_width=True)

            with col2:
                with st.spinner('Analizando la imagen...'):
                    texto, imagen_analizada_cv = analizar_imagen(imagen_pil)

                    # --- Opciones de Descarga ---
                    st.subheader("3. Descargar Resultados")

                    # Convertir la imagen analizada de OpenCV a PIL y luego a bytes
                    imagen_analizada_pil = Image.fromarray(cv2.cvtColor(imagen_analizada_cv, cv2.COLOR_BGR2RGB))
                    buf = BytesIO()
                    imagen_analizada_pil.save(buf, format="PNG")
                    bytes_img = buf.getvalue()

                    st.download_button(
                        label="Descargar Imagen con Análisis",
                        data=bytes_img,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analisis.png",
                        mime="image/png"
                    )

                    st.download_button(
                        label="Descargar Texto Extraído",
                        data=texto.encode('utf-8'),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_texto.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
