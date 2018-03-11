from word2vec_platform.word2vecReader import Word2Vec
# from  gensim.models import Word2Vec
text = "donald"

model_path = "word2vec_twitter_model/word2vec_twitter_model.bin"
print("Loading the model, this can take some time...")
model = Word2Vec.load_word2vec_format(model_path, binary=True,norm_only=True)
print(len(model.syn0norm[model.vocab[text].index]))