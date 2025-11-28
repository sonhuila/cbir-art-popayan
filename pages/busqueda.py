import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import os

from extractors.normalize_features import normalize_feature_dict
from search_engine.ranking import rank_images
from search_engine.similarity import chi_square, l2_dist, hamming_dist
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

@st.cache_resource
def load_database():
    with open("data/database.json") as f:
        db = json.load(f)
    for item in db:
        for key, value in item['features'].items():
            dtype = np.uint8 if key == 'orb' else np.float32
            item['features'][key] = np.array(value, dtype=dtype)
    return db

database = load_database()
image_path_map = {item["id"]: item["image_path"] for item in database}
db_by_id = {item['id']: item for item in database}

uploaded_file = st.file_uploader("Selecciona una imagen de consulta", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Imagen de consulta", width=300)

    # 1. Procesamiento y Extracción (sin cambios)
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    raw_features = {
        "color_moments": extract_color_moments(img_cv2), "lbp_histogram": extract_lbp(img_cv2),
        "haralick_features": extract_haralick(img_cv2), "orb": extract_orb(img_cv2)
    }
    normalized_features = normalize_feature_dict(raw_features)
    
    with st.expander("Ver Características Normalizadas de la Consulta (JSON)"):
        st.json({k: v.tolist() for k, v in normalized_features.items()})

    # 2. Búsqueda con Ranking Ponderado (sin cambios)
    st.write("Calculando similitud...")
    K = 20
    distance_functions = { "color_moments": l2_dist, "lbp_histogram": chi_square, "haralick_features": l2_dist, "orb": hamming_dist }
    weights = { "color_moments": 0.50, "lbp_histogram": 0.25, "haralick_features": 0.20, "orb": 0.05 }
    results = rank_images(normalized_features, database, weights, distance_functions, top_k=K)

    # --- 3. ANÁLISIS CON UMBRAL Y CÁLCULO DE MÉTRICAS ---
    show_metrics = False
    query_item_found = None
    
    # Definimos el umbral de tolerancia para una "coincidencia exacta"
    MATCH_THRESHOLD = 0.35  # Ajusta este valor si es necesario

    if results:
        top_result_id, top_result_dist = results[0]
        
        # --- ¡CAMBIO CLAVE! ---
        # En lugar de buscar una distancia de cero, comprobamos si está por debajo del umbral
        if top_result_dist < MATCH_THRESHOLD:
            show_metrics = True
            query_item_found = db_by_id.get(top_result_id)
            if query_item_found:
                query_class = query_item_found['class']
                st.success(f"¡Coincidencia encontrada! La imagen subida es '{query_item_found['id']}' (Clase: **{query_class}**). Calculando métricas de rendimiento.")

    # 4. Mostrar Resultados (sin cambios)
    st.subheader(f"Top {K} Resultados Similares")
    cols = st.columns(5)
    for i, (img_id, dist) in enumerate(results):
        path = image_path_map.get(img_id)
        if path and os.path.exists(path):
            with cols[i % 5]:
                border_style = "border: 5px solid #28a745;" if (show_metrics and i == 0) else ""
                st.markdown(f'<div style="{border_style}">', unsafe_allow_html=True)
                st.image(path, caption=f"Dist: {dist:.4f}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # 5. Mostrar Métricas (Solo si se activaron)
    if show_metrics and query_item_found:
        st.subheader("Métricas de Rendimiento para esta Consulta")

        retrieved_ids = [item[0] for item in results[1:]]
        retrieved_items = [db_by_id.get(id) for id in retrieved_ids if db_by_id.get(id)]
        
        total_relevant_items = [item for item in database if item['class'] == query_class and item['id'] != query_item_found['id']]

        true_positives = sum(1 for item in retrieved_items if item['class'] == query_class)
        
        precision = true_positives / len(retrieved_items) if retrieved_items else 0.0
        recall = true_positives / len(total_relevant_items) if total_relevant_items else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric(label=f"Precision@{K-1}", value=f"{precision:.2%}", help="De los resultados mostrados (sin contar la propia imagen), qué porcentaje es correcto.")
        col2.metric(label=f"Recall@{K-1}", value=f"{recall:.2%}", help="De todas las imágenes correctas que existen en la DB, qué porcentaje encontramos.")
        col3.metric(label=f"F1-Score@{K-1}", value=f"{f1_score:.2%}", help="Balance entre Precision y Recall.")

        with st.expander("Detalles del cálculo"):
            st.markdown(f"...") # La lógica interna se mantiene

    elif results:
        st.info("No se calcularon métricas. La imagen subida parece ser nueva (la distancia mínima fue superior al umbral de coincidencia).")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)