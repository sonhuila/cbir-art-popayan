import streamlit as st
import cv2
import numpy as np
import os
from build_features import extract_features
from search_engine.similarity import find_similar

st.title("🎨 CBIR - Patrimonio Artístico de Popayán")

dataset_path = "dataset/wikiart"
features = np.load("features.npy")
filenames = np.load("filenames.npy")

query_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg","png","jpeg"])

if query_file:
    file_bytes = np.asarray(bytearray(query_file.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    query_img = cv2.resize(query_img, (256, 256))
    st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), caption="Imagen de consulta")

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
