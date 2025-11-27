import streamlit as st

st.set_page_config(
    page_title="CBIR - Acerca del Proyecto",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Acerca del Proyecto")
st.write("""
Este proyecto de Recuperación de Imágenes Basada en Contenido (CBIR) tiene como objetivo proporcionar una herramienta eficiente para buscar obras artísticas similares dentro de un dataset específico.

Utilizando técnicas avanzadas de procesamiento de imágenes y aprendizaje automático, el sistema extrae características visuales de las imágenes para compararlas y encontrar similitudes.

El proyecto está diseñado para facilitar la exploración y el análisis de colecciones de arte, permitiendo a los usuarios descubrir obras relacionadas basadas en atributos visuales como color, textura y formas. Esta herramienta es especialmente útil para investigadores, curadores y entusiastas del arte que buscan identificar patrones o influencias entre diferentes obras.

A través de una interfaz intuitiva desarrollada con Streamlit, los usuarios pueden subir una imagen y recibir rápidamente una lista de imágenes similares del dataset, mejorando así la experiencia de búsqueda y análisis en el ámbito del arte digital.
""")

with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)