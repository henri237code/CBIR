# CBIR
Recherche d’images par le contenu
Cette application web permet d’effectuer une recherche d’images basée sur le contenu visuel d’une image requête, sans passer par des mots-clés.
L’utilisateur peut téléverser une image depuis son appareil, choisir un descripteur d’image (GLCM, Haralick, BiT ou Concat) ainsi qu’une mesure de distance (Euclidienne, Manhattan, Tchebychev, Canberra) pour comparer cette image aux autres présentes dans la base.

L’application extrait alors automatiquement la signature numérique de l’image requête selon le descripteur choisi, puis compare cette signature à celles des images de la base, à l’aide de la mesure de distance sélectionnée.
Elle affiche ensuite les images les plus similaires, dans un classement basé sur la distance croissante.

L’interface est développée avec Streamlit, et les traitements d’image sont effectués avec OpenCV et NumPy. Les descripteurs sont stockés au préalable sous forme de fichiers .npy pour des performances optimales.
