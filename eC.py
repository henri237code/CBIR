import os
import numpy as np
from PIL import Image
from utils.descriptors import concat

image_folder = "data/images"
features = []
paths = []

for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(image_folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            vec = concat(np.array(img))
            features.append(vec)
            paths.append(img_name)
        except Exception as e:
            print(f"Erreur avec {img_name}: {e}")

os.makedirs("data/fichiers", exist_ok=True)
np.save("data/fichiers/signatures_Concat.npy", np.array(features))
np.save("data/fichiers/paths_Concat.npy", np.array(paths))

print(" Descripteurs Concat extraits et enregistrés.")
