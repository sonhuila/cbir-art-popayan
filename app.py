import time
from typing import List, Dict, Tuple
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageDraw

# Importa tu lógica de features y similitud
from build_features import extract_features
from search_engine.similarity import find_similar

# Opcional: menús con streamlit_option_menu
try:
    from streamlit_option_menu import option_menu
    HAS_MENU = True
except ModuleNotFoundError:
    HAS_MENU = False

# ---------- Configuración básica ----------
st.set_page_config(
    page_title="Sistema CBIR para Obras Artísticas",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATASET_PATH = "dataset/wikiart"
FEATURES_PATH = "features.npy"
FILENAMES_PATH = "filenames.npy"
TOP_N = 10

# ---------- Cargar estilos ----------
def load_css():
    try:
        with open("assets/styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("No se encontró assets/styles.css.")

load_css()

# ---------- Caché para el índice ----------
@st.cache_resource(show_spinner=True)
def load_index() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(FILENAMES_PATH):
        raise FileNotFoundError("No se encontró features.npy o filenames.npy.")
    features = np.load(FEATURES_PATH)
    filenames = np.load(FILENAMES_PATH)
    return features, filenames

# ---------- Componente UI ----------
def _placeholder_image(w=480, h=320, bg="#e6e6e6"):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, w - 20, h - 20], outline="#cfcfcf", width=6)
    d.polygon([(60, h - 60), (160, 160), (260, h - 60)], outline="#bdbdbd", width=6)
    d.ellipse([(300, 90), (340, 130)], outline="#bdbdbd", width=6)
    return img

def _result_card(name: str, dist: float, img_bgr: np.ndarray, idx: int):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)
    st.markdown(
        f"""
        <div class="card-body">
            <div class="card-title">Nombre: <strong>{os.path.basename(name)}</strong></div>
            <div class="card-text">Distancia: <strong>{dist:.3f}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Usar st.button para capturar clic
    if st.button("Ver detalles", key=f"detalle_{idx}", use_container_width=True):
        st.session_state.detalle = {
            "nombre": os.path.basename(name),
            "dist": dist,
            "imagen": img_rgb
        }
        st.session_state.mostrar_detalle = True
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Estructura UI ----------
st.markdown(
    """
    <header class="topbar" role="banner">
        <div class="brand">Sistema CBIR para Obras Artísticas</div>
    </header>
    """,
    unsafe_allow_html=True,
)

if HAS_MENU:
    st.markdown('<div class="subbar">', unsafe_allow_html=True)
    selected = option_menu(
        None,
        ["Inicio", "Acerca del Proyecto", "Dataset", "Contacto"],
        icons=["house", "info-circle", "collection", "envelope"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "var(--surface)"},
            "nav-link": {"font-weight": "600", "color": "var(--text)", "padding": "10px 22px", "border-radius": "0"},
            "nav-link-selected": {"background-color": "var(--surface)", "border-bottom": "3px solid var(--primary)"},
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    t_inicio, t_about, t_dataset, t_contact = st.tabs(["Inicio", "Acerca del Proyecto", "Dataset", "Contacto"])
    selected = "Inicio"

# ---------- Páginas ----------
def page_inicio():
    st.markdown(
        """
        <section class="hero" style="background-image:url('assets/hero.jpg');">
            <div class="hero-overlay">
                <div class="hero-title">
                    Sube una imagen de una obra para encontrar otras visualmente similares.
                </div>
                <div class="hero-subtitle">
                    Nuestro sistema analiza características visuales como color, textura y composición para conectarte con obras de arte relacionadas.
                </div>
                <a href="#busqueda" class="btn btn-primary">Empezar búsqueda</a>
            </div>
        </section>
        """, unsafe_allow_html=True
    )
    st.markdown('<div id="busqueda"></div>', unsafe_allow_html=True)

    # Cargar el índice de features
    try:
        features, filenames = load_index()
    except Exception as e:
        st.error(f"No fue posible cargar el índice: {e}")
        return

    # UI de búsqueda
    with st.container(border=True):
        uploaded = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown('<div class="hint">Formatos: JPG, PNG. Tamaño ≤ 5MB.</div>', unsafe_allow_html=True)
        ph = _placeholder_image()
        query_img_bgr = None

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            query_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if query_img_bgr is None:
                st.error("No se pudo decodificar la imagen.")
            else:
                preview_rgb = cv2.cvtColor(cv2.resize(query_img_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
                st.image(preview_rgb, caption="Imagen consultada", use_container_width=True)
        else:
            st.image(ph, caption="Vista previa", use_container_width=True)

        do_search = st.button("🔎 Buscar similitudes", use_container_width=True, type="primary")

    st.markdown(
        """
        <div class="howto">
            <div class="howto-title">¿Cómo funciona?</div>
            <ul>
                <li>Sube una fotografía de cualquier obra artística</li>
                <li>Extraemos características visuales</li>
                <li>Calculamos similitud con las obras del dataset</li>
                <li>Mostramos las más similares encontradas</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Lógica de búsqueda real
    if do_search:
        if query_img_bgr is None:
            st.warning("Primero sube una imagen para realizar la búsqueda.")
            return
        with st.spinner("Analizando imagen y buscando similitudes..."):
            try:
                q_feat = extract_features(query_img_bgr)
                results = find_similar(q_feat, features, filenames, top_n=TOP_N)
                st.session_state.results = results
            except Exception as e:
                st.error(f"Error durante la búsqueda: {e}")
                return

    # Mostrar grid resultados
    if st.session_state.get("results"):
        st.markdown("### Resultados de obras similares encontradas:")
        results = st.session_state.results
        cols_per_row = 5
        for i in range(0, min(len(results), TOP_N), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, (name, dist) in enumerate(results[i:i + cols_per_row]):
                with cols[idx]:
                    img_path = os.path.join(DATASET_PATH, name)
                    if os.path.exists(img_path):
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is not None:
                            _result_card(name, dist, img_bgr, idx=(i+idx))
                        else:
                            st.warning(f"No se pudo leer la imagen: {img_path}")
                            st.image(_placeholder_image(), use_container_width=True)
                    else:
                        st.warning(f"No existe la ruta de imagen: {img_path}")
                        st.image(_placeholder_image(), use_container_width=True)

    # Detalle obra (debajo del grid)
    if st.session_state.get("mostrar_detalle"):
        detalle = st.session_state.get("detalle", {})
        st.markdown("### Detalle de la obra seleccionada")
        if detalle.get("imagen") is not None:
            st.image(detalle["imagen"], caption=detalle.get("nombre","Obra"), use_container_width=True)
            st.write(f"Distancia: {detalle.get('dist',0):.3f}")
        if st.button("Cerrar detalle", key="cerrar_detalle"):
            st.session_state.mostrar_detalle = False

def page_about():
    st.markdown("## Acerca del Proyecto")
    st.write(
        "Este sistema CBIR (Content-Based Image Retrieval) permite consultar imágenes de obras artísticas y "
        "recuperar las más similares a partir de sus características visuales."
    )
    st.markdown("### Flujo general")
    st.markdown(
        """
        1. Preprocesamiento de imágenes
        2. Extracción de características (CNN, color, textura)
        3. Indexación (FAISS/Annoy)
        4. Consulta y ranking
        5. Visualización
        """
    )

def page_dataset():
    st.markdown("## Dataset")
    st.write("Incluye tus fuentes, licencias y estructura. Como ejemplo, mostramos un pequeño esquema.")
    st.markdown(
        """
        - Obras: id, título, autor, año, técnica, museo, url_imagen
        - Features: id_obra, vector (embeddings), norma, hash
        - Índice: método (FAISS/Annoy), parámetros (dim, métrica), fecha de construcción
        """
    )

def page_contact():
    st.markdown("## Contacto")
    st.markdown(
        """
        - Universidad del Cauca – Proyecto CBIR Artístico  
        - Desarrollado por: María Paula Barrera y Ana Isabel Quira  
        - Correo: nombre.apellido@unicauca.edu.co
        """
    )

# ---------- Router ----------
if not HAS_MENU:
    with t_inicio: page_inicio()
    with t_about: page_about()
    with t_dataset: page_dataset()
    with t_contact: page_contact()
else:
    if selected == "Inicio":
        page_inicio()
    elif selected == "Acerca del Proyecto":
        page_about()
    elif selected == "Dataset":
        page_dataset()
    else:
        page_contact()

# ---------- Footer fijo ----------
st.markdown(
    """
    <footer class="footer" role="contentinfo">
        © 2025 Universidad del Cauca – Proyecto CBIR Artístico | Desarrollado por María Paula Barrera y Ana Isabel Quira
    </footer>
    """,
    unsafe_allow_html=True,
)