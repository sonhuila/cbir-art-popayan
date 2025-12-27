import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import os

from extractors.normalize_features import normalize_feature_dict, concatenate_features
from search_engine.ranking import rank_images_by_single_vector
from search_engine.similarity import l2_dist, chi_square, hamming_dist
from extractors.color_features import extract_color_moments
from extractors.texture_features import extract_lbp, extract_haralick
from extractors.keypoint_features import extract_orb

st.set_page_config(page_title="CBIR - Buscar por Imagen", page_icon="🔎", layout="wide")

with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Buscar por Imagen")
st.write("Sube una imagen para buscar obras similares en el dataset.")

# --- 1. CARGA DE DATOS OPTIMIZADA ---
@st.cache_resource
def load_database_and_vectors():
    """
    Carga la base de datos y extrae los vectores de características pre-calculados.
    """
    try:
        with open("data/database.json") as f:
            database = json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'data/database.json'.")
        return [], {}, {}

    db_vectors = []
    db_by_id = {}
    
    for item in database:
        # Asumimos que el vector concatenado está guardado bajo la clave 'concatenated_features'
        if 'features' in item:
            # Convierte la lista del JSON a un array de NumPy
            vector = np.array(item['features'], dtype=np.float32)
            db_vectors.append((item['id'], vector))
            db_by_id[item['id']] = item
        else:
            # Opcional: Advertir si un item no tiene el vector pre-calculado
            st.warning(f"El item con ID {item.get('id', 'desconocido')} no tiene un vector de características pre-calculado.")

    return db_vectors, db_by_id

# Carga los datos una sola vez
db_concatenated_vectors, db_by_id = load_database_and_vectors()


uploaded_file = st.file_uploader("Selecciona una imagen de consulta", type=["jpg", "jpeg", "png"])

if uploaded_file and db_concatenated_vectors:
    st.image(uploaded_file, caption="Imagen de consulta", width=300)

    # --- 2. PROCESAMIENTO SOLO PARA LA IMAGEN DE CONSULTA ---
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Extrae, normaliza y concatena características solo para la imagen nueva
    raw_features = {
        "color_moments": extract_color_moments(img_cv2), 
        "lbp_histogram": extract_lbp(img_cv2),
        "haralick_features": extract_haralick(img_cv2), 
        "orb": extract_orb(img_cv2)
    }
    normalized_features = normalize_feature_dict(raw_features)
    query_vector = concatenate_features(normalized_features)
    
    # --- 3. BÚSQUEDA Y RANKING ---
    st.write("Calculando similitud...")
    K = 20
    
    # Usa la función de ranking con el vector de consulta y los vectores pre-calculados
    results1 = rank_images_by_single_vector(query_vector, db_concatenated_vectors, l2_dist, top_k=K)
    results2 = rank_images_by_single_vector(query_vector, db_concatenated_vectors, chi_square, top_k=K)
    results3 = rank_images_by_single_vector(query_vector, db_concatenated_vectors, hamming_dist, top_k=K)

    # --- 4. MOSTRAR RESULTADOS ---
    st.header("Resultados de la Búsqueda con L2 Distance")
    if not results1:
        st.warning("No se encontraron resultados. Asegúrate de que la forma de los vectores coincida.")
    else:
        cols = st.columns(5)
        for i, (dist, item_id) in enumerate(results1):
            with cols[i % 5]:
                item = db_by_id[item_id]
                st.image(item["image_path"], caption=f"Dist: {dist:.4f}")
    st.header("Resultados de la Búsqueda con Chi-Square")
    if not results2:
        st.warning("No se encontraron resultados. Asegúrate de que la forma de los vectores coincida.")
    else:
        cols = st.columns(5)
        for i, (dist, item_id) in enumerate(results2):
            with cols[i % 5]:
                item = db_by_id[item_id]
                st.image(item["image_path"], caption=f"Dist: {dist:.4f}")

    st.header("Resultados de la Búsqueda con Hamming Distance")
    if not results3:
        st.warning("No se encontraron resultados. Asegúrate de que la forma de los vectores coincida.")
    else:
        cols = st.columns(5)
        for i, (dist, item_id) in enumerate(results3):
            with cols[i % 5]:
                item = db_by_id[item_id]
                st.image(item["image_path"], caption=f"Dist: {dist:.4f}")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)