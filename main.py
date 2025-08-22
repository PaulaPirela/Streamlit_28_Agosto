import cv2
import pytesseract
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import os

def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen para extraer texto y colores predominantes.

    Args:
        ruta_imagen (str): La ruta al archivo de imagen.

    Returns:
        tuple: Una tupla que contiene el texto extraído y la imagen con los colores predominantes.
    """
    # --- Carga de la imagen ---
    try:
        imagen = Image.open(ruta_imagen)
        imagen_cv = cv2.imread(ruta_imagen)
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_imagen}' no fue encontrado.")
        return None, None
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None, None

    # --- Extracción de texto con OCR ---
    print("--- Extrayendo texto de la imagen... ---")
    try:
        # Es posible que necesites configurar la ruta del ejecutable de Tesseract
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Ejemplo para Windows
        texto_extraido = pytesseract.image_to_string(imagen)
        if not texto_extraido.strip():
            print("No se encontró texto en la imagen.")
            texto_extraido = "No se detectó texto."
        else:
            print("Texto extraído:")
            print(texto_extraido)
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract no está instalado o no se encuentra en la ruta del sistema.")
        print("Por favor, instala Tesseract y/o configura la ruta del ejecutable en el script.")
        return None, imagen_cv
    except Exception as e:
        print(f"Error durante el OCR: {e}")
        texto_extraido = "Error al procesar el texto."

    # --- Análisis de colores predominantes ---
    print("\n--- Analizando los colores predominantes... ---")
    try:
        # Redimensionar la imagen para un procesamiento más rápido
        pixeles = imagen_rgb.reshape((-1, 3))

        # Usar K-Means para encontrar los 5 colores más comunes
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixeles)
        colores_predominantes = kmeans.cluster_centers_.astype(int)

        print("Colores predominantes (RGB):")
        for color in colores_predominantes:
            print(f"- rgb({color[0]}, {color[1]}, {color[2]})")

        # --- Visualización de los colores ---
        altura_barra = 50
        ancho_paleta = 300
        paleta = np.zeros((altura_barra, ancho_paleta, 3), np.uint8)
        paso = ancho_paleta // len(colores_predominantes)

        for i, color in enumerate(colores_predominantes):
            paleta[:, i * paso:(i + 1) * paso, :] = color

        # Convertir la paleta a un formato que se pueda mostrar con la imagen original
        paleta_rgb = cv2.cvtColor(paleta, cv2.COLOR_RGB2BGR)

        # Añadir un borde a la paleta para separarla de la imagen
        paleta_con_borde = cv2.copyMakeBorder(paleta_rgb, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Redimensionar la imagen original para mantener una visualización consistente
        escala = 300 / imagen_cv.shape[1]
        nueva_altura = int(imagen_cv.shape[0] * escala)
        imagen_redimensionada = cv2.resize(imagen_cv, (300, nueva_altura))

        # Combinar la imagen original con la paleta de colores
        imagen_final = np.vstack((imagen_redimensionada, paleta_con_borde))

    except Exception as e:
        print(f"Error durante el análisis de color: {e}")
        imagen_final = imagen_cv # Devolver la imagen original si falla el análisis de color

    return texto_extraido, imagen_final

if __name__ == "__main__":
    # --- Entrada del usuario ---
    ruta_input = input("Por favor, introduce la ruta de la imagen (PNG, JPG, JPEG): ")

    if not os.path.exists(ruta_input):
        print("La ruta especificada no existe.")
    else:
        # --- Procesamiento y Salida ---
        texto, imagen_analizada = analizar_imagen(ruta_input)

        if imagen_analizada is not None:
            # Mostrar la imagen con el análisis
            cv2.imshow('Análisis de Imagen', imagen_analizada)
            print("\n--- Se ha generado una ventana con la imagen y su paleta de colores. ---")
            print("Presiona cualquier tecla para cerrar la ventana.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Guardar los resultados
            nombre_base, extension = os.path.splitext(os.path.basename(ruta_input))
            ruta_salida_imagen = f"{nombre_base}_analizada.png"
            ruta_salida_texto = f"{nombre_base}_analisis.txt"

            cv2.imwrite(ruta_salida_imagen, imagen_analizada)
            with open(ruta_salida_texto, 'w', encoding='utf-8') as f:
                f.write("--- ANÁLISIS DE LA IMAGEN ---\n\n")
                f.write("Texto Extraído:\n")
                f.write(texto)
                f.write("\n\n----------------------------\n")

            print(f"\nResultados guardados en '{ruta_salida_imagen}' y '{ruta_salida_texto}'.")
