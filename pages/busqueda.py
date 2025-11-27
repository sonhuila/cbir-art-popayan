import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Importa tus módulos CBIR
from extractors.build_feature_vector import build_feature_vector
from extractors.color_features import extract_color_moments
from extractors.texture_features import extract_lbp, extract_haralick
from extractors.keypoint_features import extract_orb

from search_engine.similarity import chi_square, bhattacharyya, l2_dist, l1_dist, cosine_dist, hamming_dist
from search_engine.ranking import rank_images

import json
import os

st.set_page_config(
    page_title="CBIR - Buscar por Imagen",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar header y estilos
with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Buscar por Imagen")
st.write("Sube una imagen para buscar obras similares en el dataset.")

# ===========================
# CARGAR BASE DE DATOS
# ===========================

@st.cache_resource
def load_database():
    with open("data/database.json") as f:
        db = json.load(f)
    # Convertir vectores a numpy arrays
    for item in db:
        features = item["features"]
        if features.get("orb") is not None:
            features["orb"] = np.array(features["orb"])
        features["color_moments"] = np.array(features["color_moments"])
        features["lbp_histogram"] = np.array(features["lbp_histogram"])
        features["haralick_features"] = np.array(features["haralick_features"])
    return db

database = load_database()
image_path_map = {item["id"]: item["image_path"] for item in database}

# ===========================
# SUBIR IMAGEN QUERY
# ===========================

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Imagen subida", width='content')

    # 1. Abrir con PIL
    img = Image.open(uploaded_file).convert("RGB")

    # 2. Convertir de PIL a NumPy array
    img_np = np.array(img)

    # 3. Convertir de RGB a BGR para OpenCV si tus extractores lo requieren
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ===========================
    # 1. EXTRAER CARACTERÍSTICAS
    # ===========================
    raw_feats = {
        "color_moments": extract_color_moments(img_cv2),
        "lbp_histogram": extract_lbp(img_cv2),
        "haralick_features": extract_haralick(img_cv2),
        "orb": extract_orb(img_cv2)
    }

    st.write("Características extraídas:", raw_feats)

    # ===========================
    # 2. VECTOR FINAL
    # ===========================
    query_vector = build_feature_vector(raw_feats)
    st.write("Vector de características normalizado:", query_vector)

    # ===========================
    # 3. BÚSQUEDA SIMPLIFICADA
    # ===========================
    st.write("Calculando similitud...")

    # Computa distancia L2 entre el query y cada imagen en la base
    scores = []
    for item in database:
        db_vector = build_feature_vector(item["features"])
        dist = l2_dist(query_vector, db_vector)
        scores.append((item["id"], dist))

    scores.sort(key=lambda x: x[1])
    results = scores[:20]

    # ===========================
    # 4. MOSTRAR RESULTADOS
    # ===========================
    st.subheader("Resultados similares")
    if not results:
        st.warning("No se encontraron resultados. Intenta con otra imagen o verifica la base de datos.")
    else:
        cols = st.columns(4)
        idx = 0
        for img_id, dist in results:
            path = image_path_map.get(img_id)
            if path and os.path.exists(path):
                with cols[idx % 4]:
                    st.image(path, caption=f"ID: {img_id} | Similitud: {dist:.4f}", width='content')
            else:
                with cols[idx % 4]:
                    st.warning(f"No se encontró la ruta para la imagen con ID: {img_id}")
            idx += 1

    # ===========================
    # OPCIONAL: RANKING AVANZADO
    # ===========================
    # distance_functions = {
    #     "color_moments": l2_dist,
    #     "lbp_histogram": chi_square,
    #     "haralick_features": cosine_dist,
    #     "orb": hamming_dist
    # }
    # weights = {
    #     "color_moments": 0.2,
    #     "lbp_histogram": 0.2,
    #     "haralick_features": 0.2,
    #     "orb": 0.4
    # }
    # ranking = rank_images(raw_feats, database, weights, distance_functions, top_k=20)
    # st.write("Ranking avanzado:", ranking)

# Footer
with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)