# import emoji
import numpy as np
import re
from word2vec_platform.word2vecReader import Word2Vec

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess_tweet(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
# def char_is_emoji(character):
#     return character in emoji.UNICODE_EMOJI
#
#
# def text_has_emoji(text):
#     for character in text:
#         if character in emoji.UNICODE_EMOJI:
#             return True
#     return False
# def extract_emojis(str):
#   return ''.join([c for c in str if c in emoji.UNICODE_EMOJI])

# df = pd.read_csv("tweets.csv")
# df["has.emoji"]=None
# emo = 0
# for i,row in df.iterrows():
#     if text_has_emoji(row['text']):
#         df.set_value(i,'has.emoji',1)
#         emo+=1
#     else:
#         df.set_value(i, 'has.emoji', 0)
#
# print(float(emo)/len(df))
# df.to_csv("tweets_updated.csv")



def add_extra_attributes(df):
    df["text_length"]=None
    df["popularity"]=None
    for index, row in df.iterrows():
        df.set_value(index, "text_length", len(preprocess_tweet(row["text"])))
        if float(row["user.followers_count"]) == 0:
            value = -1
        else:
            value = (0.1 * row["favorite_count"] + 0.9 * row["retweet_count"]) / float(row["user.followers_count"])
        df.set_value(index, "popularity", value)
    df["text_length"] = df["text_length"].astype(float)
    df["popularity"] = df["popularity"].astype(float)
    return df


def add_treatment_flag_for_dots(df, treatments):
    treatment_names={"?":"contains_dot","!":"contains_dot",".":"contains_dot"}
    df[treatment_names[treatments[0]]]=None
    for index, row in df.iterrows():
        for treatment in treatments:
            if treatment in preprocess_tweet(row["text"]):
                value=1
            else:
                value=0
            df.set_value(index, treatment_names[treatment], value)
    return df

def add_treatment_flag_for_image(df):
    df["images"]=None
    for index, row in df.iterrows():
        urls = re.findall("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", row["text"])
        if urls:
            value=1
        else:
            value=0
        df.set_value(index, "images", value)
    return df

def get_averages_for_tests_dots(df):
    model_path = "../word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True, norm_only=True)
    has = df[df.contains_dot==1]
    no = df[df.contains_dot==0]
    has["text_vector"]=None
    no["text_vector"]=None
    has = has.reset_index(drop=True)
    no = no.reset_index(drop=True)
    for index,row in has.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector =np.zeros(400)
        d=0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d+=1
        vector=vector/d
        has.set_value(index,"text_vector",vector)
    for index,row in no.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector =np.zeros(400)
        d=0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d+=1
        vector=vector/d
        no.set_value(index,"text_vector",vector)
    return has[["text_vector","popularity"]],no[["text_vector","popularity"]]

def get_averages_for_tests_images(df):
    model_path = "../word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True, norm_only=True)
    has = df[df.images==1]
    no = df[df.images==0]
    has["text_vector"] = None
    no["text_vector"] = None
    has = has.reset_index(drop=True)
    no = no.reset_index(drop=True)
    for index, row in has.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector = np.zeros(400)
        d = 0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d += 1
        vector = vector / d
        has.set_value(index, "text_vector", vector)
    for index, row in no.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector = np.zeros(400)
        d = 0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d += 1
        vector = vector / d
        no.set_value(index, "text_vector", vector)
    return has[["text_vector", "popularity"]], no[["text_vector", "popularity"]]