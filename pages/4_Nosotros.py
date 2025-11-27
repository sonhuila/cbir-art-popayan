import streamlit as st

st.set_page_config(
    page_title="CBIR - Nosotros",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Nosotros")
st.write("""
Somos un equipo apasionado por la tecnología y el arte, comprometidos en desarrollar soluciones innovadoras para la recuperación de imágenes basadas en contenido.

Nuestro objetivo es facilitar el acceso y la exploración de colecciones artísticas mediante herramientas avanzadas que combinan procesamiento de imágenes y aprendizaje automático.

Creemos en el poder del arte para conectar personas y culturas, y estamos dedicados a crear plataformas que permitan a los usuarios descubrir y apreciar obras de arte de manera más profunda e interactiva.

A través de este proyecto, esperamos contribuir al campo del arte digital y apoyar a investigadores, curadores y entusiastas del arte en su búsqueda de conocimiento y apreciación artística.
""")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)