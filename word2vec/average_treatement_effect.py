from word2vec_platform.knn import knn
import numpy as np
from functools import partial
from multiprocessing import Pool


def calcualte_te(treated_features, not_treated_features, search, args):
    index,row = args
    sample_popularity = treated_features.get_value(index,"popularity")
    nn = knn(row, search, 5,"L2")
    averaged_nt_popularity = []
    for i in nn:
        averaged_nt_popularity.append(not_treated_features.get_value(i, 'popularity'))
    averaged_nt_popularity_score = np.mean(averaged_nt_popularity)
    res = sample_popularity - averaged_nt_popularity_score
    return res


def get_average_treatment_effect(not_treated_features,treated_features):
    print("calculation average treatment effect")
    search = not_treated_features["text_vector"]
    search=search.as_matrix()
    treatment_features_matrix = treated_features["text_vector"].as_matrix()
    elements = [(index,row) for index,row in enumerate(treatment_features_matrix)]
    f=partial(calcualte_te, treated_features, not_treated_features, search)
    with Pool(processes=15) as pool:
        TE = pool.map(f,elements)
        return np.mean(TE)
