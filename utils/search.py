import os
import numpy as np
from PIL import Image
from utils.distances import distances_dict


def rechercher_similaires(img_vec, features, paths, distance_name, top_k=5):
    results = []
    for i in range(len(features)):
        vec = features[i]
        path = paths[i]
        dist = distances_dict[distance_name](img_vec, vec)
        results.append((path, dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

