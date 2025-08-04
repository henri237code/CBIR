import os
import numpy as np
from utils.descriptors import glcm_rgb  
from PIL import Image

image_folder = "data/images"
fichiers = []
image_names = []

for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(image_folder, img_name)
        try:
            vec = glcm_rgb(img_path)
            fichiers.append(vec)
            image_names.append(img_name)
        except Exception as e:
            print(f"Erreur pour {img_name} :", e)

os.makedirs("data/fichiers", exist_ok=True)
np.save("data/fichiers/signatures_GLCM.npy", np.array(fichiers))
np.save("data/fichiers/paths_GLCM.npy", np.array(image_names))


print(" Descripteurs enregistrés avec succès.")
