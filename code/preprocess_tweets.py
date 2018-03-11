# import emoji
import pandas as pd
import numpy as np
import re
from word2vec_platform.word2vecReader import Word2Vec
from sklearn.decomposition import PCA
import pscore

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
            value = (0.1 * row["favorite_count"] + 0.9 * row["retweet_count"]) / np.log(float(row["user.followers_count"]))
        df.set_value(index, "popularity", value)
    df["text_length"] = df["text_length"].astype(float)
    df["popularity"] = df["popularity"].astype(float)
    df['hour'] = pd.to_datetime(df['created_at']).dt.hour
    df['time_of_day'] = 0
    df['time_of_day'][(df['hour'] > 7) & (df['hour'] < 12)] = 1
    df['time_of_day'][(df['hour'] > 12) & (df['hour'] < 17)] = 2
    df['time_of_day'][(df['hour'] > 17) & (df['hour'] < 23)] = 3
    return df


def add_treatment_flag_for_dots(df, treatments):
    treatment_names={"?":"contains_dot","!":"contains_dot",".":"contains_dot"}
    df[treatment_names[treatments[0]]]=None
    for index, row in df.iterrows():
        for treatment in treatments:
            if treatment in preprocess_tweet(row["text"]):
                value=1
                break
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

def get_averages_for_tests(df,treatment):
    model_path = "word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True, norm_only=True)
    df["text_vector"]=None
    matrix=[]
    has_treatment = {}
    no_treatment = {}

    pca = PCA(n_components=85,random_state=1)
    for index,row in df.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector =np.zeros(400)
        d=0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d+=1
        if d!=0:
            vector = vector / d
            put="OK"
            matrix.append(vector)
        else:
            put="E"
        df.set_value(index,"text_vector",put)
    df = df[df.text_vector != "E"]
    df=df.reset_index(drop=True)
    matrix_reduced=pca.fit_transform(matrix)
    index_for_treated=0
    index_for_non_treated=0
    has_treatment_popularity={}
    no_treatment_popularity={}
    for index,row in df.iterrows():
        if row[treatment]==1:
            has_treatment_popularity[index_for_treated]=row["popularity"]
            has_treatment[index_for_treated]={}
            length = row["text_length"]
            time = row["time_of_day"]
            vector = matrix_reduced[index]
            has_treatment[index_for_treated]["length"]=length
            has_treatment[index_for_treated]["time_of_day"]=time
            for i, data_point in enumerate(vector):
                has_treatment[index_for_treated][i] = data_point
            index_for_treated+=1

        else:
            no_treatment_popularity[index_for_non_treated]=row["popularity"]
            no_treatment[index_for_non_treated]={}
            length = row["text_length"]
            vector = matrix_reduced[index]
            no_treatment[index_for_non_treated]["length"] = length
            for i, data_point in enumerate(vector):
                no_treatment[index_for_non_treated][i] = data_point
            index_for_non_treated+=1
    return has_treatment_popularity,no_treatment_popularity,pd.DataFrame.from_dict(has_treatment,orient="index"),pd.DataFrame.from_dict(no_treatment,orient="index")



