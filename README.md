# Sistema CBIR para Obras de Arte

 Un sistema de Búsqueda de Imágenes por Contenido (CBIR) para encontrar obras de arte similares basado en características visuales como color, textura y forma, implementado con Python y Streamlit.

## Características Principales

- Extracción de 4 tipos de características visuales: Momentos de Color, LBP, Haralick y ORB.
- Motor de búsqueda con **ranking ponderado** para ajustar la importancia de cada característica.
- Interfaz de usuario web interactiva construida con Streamlit.
- Cálculo automático de métricas de rendimiento si la imagen de consulta existe en la base de datos.
- Arquitectura modular y extensible.

## Estructura del Proyecto

```
cbir-art-popayan/
├── assets/             # Archivos para la UI (CSS, HTML)
├── data/               # Contiene el dataset y la base de datos generada
├── dataset/            # Imágenes del dataset organizadas en carpetas por clase
├── extractors/         # Módulos para extraer las características de las imágenes
├── search_engine/      # Lógica del motor de búsqueda (ranking, similitud)
├── pages/              # Páginas de la aplicación Streamlit
├── build_database.py   # Script para pre-procesar el dataset y crear database.json
└── app.py    # Punto de entrada principal de la aplicación Streamlit
```

## Requisitos

- Python 3.8+
- Las librerías listadas en `requirements.txt`

## Instalación

1.  Clona el repositorio:
    ```bash
    git clone [URL de tu repositorio]
    cd cbir-art-popayan
    ```

2.  Crea un entorno virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para usar la aplicación, primero debes construir la base de datos de características y luego ejecutar la aplicación web.

1.  **Construir la Base de Datos:**
    Asegúrate de que tus imágenes estén organizadas en subcarpetas por clase dentro de `dataset/wikiart/`. Luego, ejecuta el siguiente comando desde la raíz del proyecto:
    ```bash
    python build_database.py
    ```
    Esto creará el archivo `data/database.json` con las características pre-procesadas de todas las imágenes.

2.  **Ejecutar la Aplicación Web:**
    ```bash
    streamlit run app.py
    ```
    Abre tu navegador y ve a la dirección URL que te indica Streamlit (usualmente `http://localhost:8501`).

---