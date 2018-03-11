import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



def get_matches(treated_df, non_treated_df,k, scaler=True):

    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    if scaler == True:
        scaler = StandardScaler()
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',n_jobs=15).fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x)
    return indices

def get_ATE(treated_popularity,not_treated_popularity,treated_df, non_treated_df,indices):
    TE=[]
    treated_ids = treated_df.index.values
    not_treated_ids = non_treated_df.index.values
    for treated_index, relevant_indices in enumerate(indices):
        treated_id = treated_ids[treated_index]
        sample_popularity = treated_popularity[treated_id]["popularity"]
        counterfactual_populrity=[]
        for i in relevant_indices:
            not_treated_id=not_treated_ids[i]
            counterfactual_populrity.append(not_treated_popularity[not_treated_id]["popularity"])
        if (sample_popularity - np.mean(counterfactual_populrity)) > -0.05:
            TE.append(sample_popularity-np.mean(counterfactual_populrity))
        # TE.append(sample_popularity - np.mean(counterfactual_populrity))
    return np.mean(TE)

def get_average_treatment_effect(treated_features,not_treated_features,not_treated_popularity,treated_popularity,n_neighbors):
    print("calculation average treatment effect")
    print("in KNN")
    matches = get_matches(treated_features, not_treated_features, n_neighbors)
    print("in ATE calculation")
    return get_ATE(treated_popularity,not_treated_popularity,treated_features,not_treated_features,matches)

    # search = not_treated_features[keys[1:]]
    # search=search.as_matrix()
    # treatment_features_matrix = treated_features[keys[1:]].as_matrix()
    # elements = [(index,row) for index,row in enumerate(treatment_features_matrix)]
    # f=partial(calcualte_te, treated_features, not_treated_features, treated_popularity, not_treated_popularity, search)
    # with Pool(processes=15) as pool:
    #     TE = pool.map(f,elements)
    #     return np.mean(TE)



# def calcualte_te(treated_features, not_treated_features, treated_popularity, not_treated_popularity, search, args):
#     index,row = args
#     user_id = treated_features.get_value(index, 'user.id')
#     sample_popularity = treated_popularity[('popularity', 'mean')][user_id]
#     if not not_treated_popularity.get(user_id, False):
#         nn = knn(row, search, 5)
#         user_ids = []
#         for i in nn:
#             user_ids.append(not_treated_features.get_value(i, 'user.id'))
#         averaged_nt_popularity = np.mean([not_treated_popularity[('popularity', 'mean')][i] for i in user_ids])
#         res = sample_popularity - averaged_nt_popularity
#     else:
#         res = sample_popularity - not_treated_popularity[user_id]
#     return res
