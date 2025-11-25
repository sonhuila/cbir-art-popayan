import streamlit as st
import cv2
import numpy as np
import os
from build_features import extract_features
from search_engine.similarity import find_similar

# Configuración de la página
st.set_page_config(
    page_title="CBIR - Patrimonio Artístico de Popayán",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 CBIR - Patrimonio Artístico de Popayán")

# Barra de navegación usando tabs nativos de Streamlit
tab_inicio, tab_busqueda, tab_acerca = st.tabs(["🏠 Inicio", "🔍 Búsqueda", "ℹ️ Acerca de"])

dataset_path = "dataset/wikiart"

# Función para cargar datos con cache
@st.cache_data
def load_features():
    return np.load("features.npy"), np.load("filenames.npy")

# Pestaña de Inicio
with tab_inicio:
    st.header("Bienvenido al Sistema de Búsqueda de Imágenes por Contenido")
    st.markdown("""
    Este sistema te permite buscar imágenes similares basándose en las características visuales 
    del patrimonio artístico de Popayán.
    
    ### ¿Cómo funciona?
    1. Ve a la pestaña **🔍 Búsqueda**
    2. Sube una imagen de referencia
    3. El sistema encontrará las imágenes más similares en nuestra base de datos
    
    ### Características del sistema:
    - **Análisis de color**: Detecta momentos de color (media, desviación, asimetría)
    - **Análisis de textura**: Utiliza LBP y características de Haralick
    - **Detección de puntos clave**: Emplea descriptores ORB
    """)
    
    features, filenames = load_features()
    st.info(f"📊 Base de datos actual: **{len(filenames)}** imágenes disponibles")

# Pestaña de Búsqueda
with tab_busqueda:
    st.header("Búsqueda de Imágenes Similares")
    
    query_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg","png","jpeg"])

    if query_file:
        file_bytes = np.asarray(bytearray(query_file.read()), dtype=np.uint8)
        query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        query_img = cv2.resize(query_img, (256, 256))
        st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), caption="Imagen de consulta")

        features, filenames = load_features()
        q_feat = extract_features(query_img)
        results = find_similar(q_feat, features, filenames, top_n=5)

        st.subheader("🖼️ Imágenes más similares:")
        cols = st.columns(5)
        for i, (name, dist) in enumerate(results):
            img_path = os.path.join(dataset_path, name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cols[i].image(img, caption=f"{name}\nDistancia: {dist:.2f}")

# Pestaña Acerca de
with tab_acerca:
    st.header("Acerca de este proyecto")
    st.markdown("""
    ### CBIR - Content-Based Image Retrieval
    
    Sistema de recuperación de imágenes basado en contenido para el patrimonio artístico de Popayán.
    
    #### Tecnologías utilizadas:
    - **Streamlit**: Framework de interfaz de usuario
    - **OpenCV**: Procesamiento de imágenes
    - **scikit-image**: Extracción de características de textura
    - **NumPy/SciPy**: Cálculos numéricos y medidas de similitud
    
    #### Métodos de extracción de características:
    | Tipo | Método | Descripción |
    |------|--------|-------------|
    | Color | Momentos de color | Media, desviación estándar y asimetría por canal RGB |
    | Textura | LBP | Local Binary Patterns - patrones binarios locales |
    | Textura | Haralick | Características GLCM (contraste, homogeneidad, energía, correlación) |
    | Forma | ORB | Oriented FAST and Rotated BRIEF - puntos clave |
    
    ---
    *Desarrollado para el patrimonio artístico de Popayán* 🇨🇴
    """)
