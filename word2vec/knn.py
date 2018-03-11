import math
from scipy.spatial.distance import mahalanobis

def norm(v1,v2):
    tmp = v2-v1
    dist = [(a) ** 2 for a in tmp]
    dist = math.sqrt(sum(dist))
    return dist

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)

def metric(v1,v2,type,VI=None):
    if type=="L2":
        return norm(v1,v2)
    if type=="cos":
        return cosine_similarity(v1,v2)
    if type=="mahala":
        if VI is None:
            print("calculate VI!!")
            raise Exception
        return mahalanobis(v1,v2,VI)

def knn(point,data,k,type):
    nearest_neighbors=[]
    for index,sample in enumerate(data):
        nearest_neighbors.append((index,metric(sample,point,type)))
    return [y[0] for y in sorted(nearest_neighbors,key=lambda x:x[1])[:k]]