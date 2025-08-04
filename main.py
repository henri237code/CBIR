import streamlit as st 
import os
import numpy as np
from PIL import Image
import tempfile

from utils.search import rechercher_similaires
from utils.descriptors import glcm_rgb as glcm, haralick_feat, bitdesc_feat, concat
from utils.distances import distances_dict

st.set_page_config(page_title="Recherche d'images par contenu", layout="wide")
st.title("CBIR - Recherche d'images par le contenu")

chemin_dossier_images = "data/images"
chemin_dossier_features = "data/fichiers"

uploaded_image = st.file_uploader("Téléverse une image pour la recherche", type=["png", "jpg", "jpeg"])
descripteur_nom = st.selectbox("Choisis un descripteur", ["GLCM", "Haralick", "BiT", "Concat"])
distance_nom = st.selectbox("Choisis une mesure de distance", list(distances_dict.keys()))
n_resultats = st.slider("Nombre d'images similaires à afficher", 1, 20, 5)

if uploaded_image:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Image requête", use_column_width=True)

    if st.button("Lancer la recherche"):
        st.info("Recherche en cours...")

        
        if descripteur_nom == "GLCM":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                query_sig = glcm(tmp.name)
        elif descripteur_nom == "Haralick":
            query_sig = haralick_feat(np.array(image))
        elif descripteur_nom == "BiT":
            query_sig = bitdesc_feat(np.array(image))
        else:
            query_sig = concat(np.array(image))

        
        features_path = os.path.join(chemin_dossier_features, f"signatures_{descripteur_nom}.npy")
        images_path = os.path.join(chemin_dossier_features, f"paths_{descripteur_nom}.npy")
        
        try:
            features = np.load(features_path)
            chemins_images = np.load(images_path)
        except:
            st.error("Les fichiers .npy sont introuvables. Veuillez exécuter l'extraction des signatures.")
            st.stop()

        
        resultats = rechercher_similaires(query_sig, features, chemins_images, distance_nom, top_k=n_resultats)

        st.subheader("Résultats de la recherche")
        cols = st.columns(n_resultats)
        for i, (path, dist) in enumerate(resultats):
            with cols[i]:
                st.image(os.path.join(chemin_dossier_images, path), caption=f"{path}\nDist: {dist:.2f}", use_container_width=True)
