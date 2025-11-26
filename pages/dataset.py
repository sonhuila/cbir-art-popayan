import streamlit as st
from streamlit_navigation_bar import st_navbar

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["Inicio", "Acerca del Proyecto", "Dataset", "Nosotros"]
urls = {"Inicio": "/pages/inicio.py",
        "Acerca del Proyecto": "/pages/acerca_del_proyecto.py",
        "Dataset": "/pages/dataset.py",
        "Nosotros": "/pages/nosotros.py"}
styles = {"div": {"max-width": "35rem"}}

page = st_navbar(pages, urls=urls, styles=styles)

st.header(page)
