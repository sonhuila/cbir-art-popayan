import streamlit as st

st.set_page_config(
    page_title="CBIR - Inicio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar header y estilos
with open("assets/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Bienvenido a CBIR - Recuperación de Imágenes Basada en Contenido")
st.write("""
Este sistema permite buscar obras artísticas similares en un dataset utilizando la técnica de Recuperación de Imágenes Basada en Contenido (CBIR).

Utiliza el menú de la barra lateral para navegar entre las funciones disponibles: búsqueda por imagen, ver el dataset, información del proyecto y más.
""")

# Footer
with open("assets/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)