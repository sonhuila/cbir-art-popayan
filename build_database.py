import os
import json
import numpy as np
from PIL import Image
import cv2

from extractors.normalize_features import normalize_feature_dict
from extractors.color_features import extract_color_moments
from extractors.texture_features import extract_lbp, extract_haralick
from extractors.keypoint_features import extract_orb


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NumpyEncoder, self).default(obj)

def get_category_from_genre(genre_str):
    genre = genre_str.lower().replace('_', ' ').strip()
    mapping = {
        "realism": "Pintura figurativa", "baroque": "Pintura figurativa", "renaissance": "Pintura figurativa", "neoclassicism": "Pintura figurativa", 
        "romanticism": "Pintura figurativa", "symbolism": "Pintura figurativa", "post-impressionism": "Pintura figurativa", "genre painting": "Pintura figurativa", 
        "religious painting": "Pintura figurativa", "portrait": "Pintura figurativa", "early renaissance": "Pintura figurativa", "high renaissance": "Pintura figurativa", 
        "mannerism late renaissance": "Pintura figurativa", "expressionism": "Pintura figurativa", "fauvism": "Pintura figurativa", 
        "abstract art": "Pintura abstracta", "abstract expressionism": "Pintura abstracta", "cubism": "Pintura abstracta", "futurism": "Pintura abstracta", 
        "suprematism": "Pintura abstracta", "constructivism": "Pintura abstracta", "minimalism": "Pintura abstracta", "op art": "Pintura abstracta", 
        "color field painting": "Pintura abstracta", "action painting": "Pintura abstracta", "analytical cubism": "Pintura abstracta", "pop art": "Pintura abstracta", 
        "landscape": "Pintura de paisaje", "cityscape": "Pintura de paisaje", "marina / seascape": "Pintura de paisaje", "pastoral": "Pintura de paisaje",
        "drawing": "Grabados y dibujos", "sketch": "Grabados y dibujos", "ink": "Grabados y dibujos", "charcoal": "Grabados y dibujos", "lithograph": "Grabados y dibujos", 
        "etching": "Grabados y dibujos", "woodcut": "Grabados y dibujos", "digital art": "Arte digital / Otros", "mixed media": "Arte digital / Otros", 
        "collage": "Arte digital / Otros", "experimental": "Arte digital / Otros", "conceptual art": "Arte digital / Otros"
    }
    return mapping.get(genre, "Categoría desconocida")

def create_database(dataset_path, output_path):
    database = []
    print(f"Iniciando procesamiento del dataset en: {dataset_path}")

    for genre_folder_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, genre_folder_name)
        if not os.path.isdir(class_path): continue

        main_category = get_category_from_genre(genre_folder_name)
        print(f"\nProcesando carpeta '{genre_folder_name}' como categoría: '{main_category}'...")

        for filename in sorted(os.listdir(class_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                print(f"  - Procesando: {filename}")

                try:
                    img_pil = Image.open(image_path).convert("RGB")
                    img_np = np.array(img_pil)
                    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    raw_features = {
                        "color_moments": extract_color_moments(img_cv2), "lbp_histogram": extract_lbp(img_cv2),
                        "haralick_features": extract_haralick(img_cv2), "orb": extract_orb(img_cv2)
                    }

                    # 2. Normalizar el diccionario de características
                    normalized_features = normalize_feature_dict(raw_features)

                    database_entry = {
                        "id": os.path.splitext(filename)[0],
                        "image_path": image_path,
                        "class": main_category,
                        "genre": genre_folder_name,
                        "features": normalized_features
                    }
                    database.append(database_entry)

                except Exception as e:
                    print(f"    -> Error procesando {filename}: {e}")

    print(f"\nProcesamiento completado. Guardando base de datos en {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(database, f, cls=NumpyEncoder, indent=4)
    print(f"¡Base de datos creada exitosamente con {len(database)} imágenes!")

if __name__ == '__main__':
    DATASET_FOLDER = 'dataset/wikiart'
    OUTPUT_JSON_PATH = 'data/database.json'
    os.makedirs('data', exist_ok=True)
    create_database(dataset_path=DATASET_FOLDER, output_path=OUTPUT_JSON_PATH)