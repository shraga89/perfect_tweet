from word2vecReader import Word2Vec
from sklearn.decomposition import PCA
import pickle

model_path = "word2vec_twitter_model/word2vec_twitter_model.bin"
print("Loading the model, this can take some time...")
model = Word2Vec.load_word2vec_format(model_path, binary=True,norm_only=True)

# X = model[model.wv.vocab]
# words = list(model.vocab)[:10]
# print(words)
# words = ['election', 'congress', 'democrat', 'trump', 'vote', 'day', 'republican', 'out', 'just', 'all', \
#          'voting', 'what', 'hillary', 'today', 'president', 'people', 'us', 'night', 'clinton', 'her',\
#          'clinton','america', 'one', 'realdonaldtrump', 'senate', 'win', 'donald', 'obama', 'american', \
#          'house', 'polling', 'world', 'white', 'years', 'year', 'country', 'state']
X = [model.syn0norm[model.vocab[i].index] for i in words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
filename = 'two_dim_model.sav'
pickle.dump(model, open(filename, 'wb'))
filename = 'words_model.sav'
pickle.dump(model, open(filename, 'wb'))