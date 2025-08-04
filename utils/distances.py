import numpy as np

def euclidienne(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def manhattan(vec1, vec2):
    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))

def tchebychev(vec1, vec2):
    return np.max(np.abs(np.array(vec1) - np.array(vec2)))

def canberra(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.sum(np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10))  # éviter division par 0

distances_dict = {
    "Euclidienne": euclidienne,
    "Manhattan": manhattan,
    "Tchebychev": tchebychev,
    "Canberra": canberra
}
