import pickle
from matplotlib import pyplot

filename = 'two_dim_model.sav'
two_dim_model = pickle.load(open(filename, 'rb'))
filename = 'words_model.sav'
words_model = pickle.load(open(filename, 'rb'))
pyplot.scatter(two_dim_model[:, 0], two_dim_model[:, 1])
words = words_model
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(two_dim_model[i, 0], two_dim_model[i, 1]))
pyplot.show()