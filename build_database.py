import os
import json
import numpy as np
from PIL import Image

# === Importar funciones extractoras ===
from extractors.color_features import extract_color_moments
from extractors.texture_features import extract_lbp, extract_haralick
from extractors.keypoint_features import extract_orb

# --- Clase especial para convertir arrays de Numpy a listas para JSON ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def create_database(dataset_path, output_path):
    """
    Recorre un dataset de imágenes, extrae sus características y las guarda en un archivo JSON.
    """
    database = []
    
    print(f"Iniciando procesamiento del dataset en: {dataset_path}")

    # Recorrer archivos de imagen en la carpeta del dataset
    for filename in sorted(os.listdir(dataset_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(dataset_path, filename)
            
            print(f"Procesando: {filename}...")

            try:
                # Abrir imagen con PIL, y convertir a RGB
                img_pil = Image.open(image_path).convert("RGB")
                img_np = np.array(img_pil)

                # Extraer características (usa el mismo formato que tus normalizadores)
                orb_features = extract_orb(img_np)
                color_moments = extract_color_moments(img_np)
                lbp_histogram = extract_lbp(img_np)
                haralick_features = extract_haralick(img_np)

                features_dict = {
                    "color_moments": color_moments,
                    "lbp_histogram": lbp_histogram,
                    "haralick_features": haralick_features,
                    "orb": orb_features
                }

                database_entry = {
                    "id": image_id,
                    "image_path": image_path, # Puede guardar el path relativo o absoluto
                    "features": features_dict
                }
                
                database.append(database_entry)

            except Exception as e:
                print(f"  -> Error procesando {filename}: {e}")

    # --- Guardar la base de datos en archivo JSON ---
    print(f"\nProcesamiento completado. Guardando base de datos en {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(database, f, cls=NumpyEncoder, indent=4)
        
    print("¡Base de datos creada exitosamente!")

if __name__ == '__main__':
    # Cambia esta ruta a tu carpeta de imágenes
    DATASET_FOLDER = 'C:\\Users\\anais\\OneDrive\\Documentos\\GitHub\\cbir-art-popayan\\dataset\\wikiart'
    OUTPUT_JSON_PATH = 'data/database.json'
    create_database(dataset_path=DATASET_FOLDER, output_path=OUTPUT_JSON_PATH)