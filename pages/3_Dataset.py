import streamlit as st

st.set_page_config(
    page_title="CBIR - Dataset",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Dataset")
st.write("""
El dataset utilizado en este proyecto de Recuperación de Imágenes Basada en Contenido (CBIR)
está compuesto por una colección diversa de obras artísticas que abarcan diferentes estilos, épocas y medios.

Este conjunto de datos ha sido cuidadosamente seleccionado para representar una amplia gama de características visuales, 
lo que permite una evaluación efectiva del sistema CBIR.

**Estructura principal ejemplo:**
- Obras: id, título, autor, año, técnica, museo, url_imagen
- Features: id_obra, vector (embeddings), norma, hash
- Índice: método (FAISS/Annoy), parámetros (dim, métrica), fecha de construcción
""")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)