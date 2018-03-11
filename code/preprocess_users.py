# import emoji
import pandas as pd
from sklearn.decomposition import PCA
import re
from word2vec_platform.word2vecReader import Word2Vec
import numpy as np
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

def convert(x):
    return pd.Series({"popularity" : x['popularity'].mean() , "text": " ".join(x['text'])})


def create_vectors_from_sentence(df,model):
    df["text_vector"] = None
    matrix = []
    for index,row in df.iterrows():
        text = row["text"]
        tokens = preprocess_tweet(text)
        vector = np.zeros(400)
        d = 0
        for token in tokens:
            if token in model.vocab:
                vector += model.syn0norm[model.vocab[token].index]
                d += 1

        if d != 0:
            vector = vector / d
            put="OK"
            matrix.append(vector)
        else:
            put = "E"
        df.set_value(index, "text_vector", put)
    df = df[df.text_vector!="E"]
    # df = df.reset_index(drop=False)
    pca = PCA(n_components=85, random_state=1)
    reduced_matrix = pca.fit_transform(matrix)
    i=0
    for index,row in df.iterrows():
        df.set_value(index, "text_vector", reduced_matrix[i])
        i+=1
    return df


def create_joined_data_sets(df_text,original_features,keys):
    original_features.reset_index(drop=True)
    data_set={}
    for index,row in original_features.iterrows():
        id = row["user.id"]
        if id in data_set:
            continue
        if id not in df_text.index:
            continue
        data_set[id] = {}
        for key in keys:
            data_set[id][key]=row[key]
        vector = df_text.get_value(id,"text_vector")
        for i,data_point in enumerate(vector):
            data_set[id][i]=data_point
    return pd.DataFrame.from_dict(data_set,orient="index")



def get_averages_for_tests_dots(df, keys):
    model_path = "word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True, norm_only=True)
    has = df[df.contains_dot==1]
    no = df[df.contains_dot==0]
    has = has.reset_index(drop=True)
    no = no.reset_index(drop=True)
    tmp_no=no.groupby(["user.id"])[["popularity","text"]].apply(convert)
    tmp_has=has.groupby(["user.id"])[["popularity","text"]].apply(convert)
    tmp_no=create_vectors_from_sentence(tmp_no,model)
    tmp_has=create_vectors_from_sentence(tmp_has,model)
    # has_treatment_data= create_joined_data_sets(tmp_has,has,keys)
    # no_treatment_data= create_joined_data_sets(tmp_no,no,keys)
    # has_treatment_data,no_treatment_data = remove_outliers(has_treatment_data,no_treatment_data)

    tmp_has = tmp_has[["popularity"]]
    tmp_no = tmp_no[["popularity"]]
    return tmp_has.to_dict(orient="index"),tmp_no.to_dict(orient="index"),has_treatment_data,no_treatment_data

def get_averages_for_tests_images(df, keys):
    model_path = "word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True, norm_only=True)
    has = df[df.images==1]
    no = df[df.images==0]
    has = has.reset_index(drop=True)
    no = no.reset_index(drop=True)
    tmp_no = no.groupby(["user.id"])[["popularity", "text"]].apply(convert)
    tmp_has = has.groupby(["user.id"])[["popularity", "text"]].apply(convert)
    tmp_no = create_vectors_from_sentence(tmp_no, model)
    tmp_has = create_vectors_from_sentence(tmp_has, model)
    has_treatment_data = create_joined_data_sets(tmp_has, has, keys)
    no_treatment_data = create_joined_data_sets(tmp_no, no, keys)
    tmp_has=tmp_has[["popularity"]]
    tmp_no=tmp_no[["popularity"]]
    return tmp_has.to_dict(orient="index"), tmp_no.to_dict(orient="index"), has_treatment_data, no_treatment_data

#
# def remove_outliers(df_treated,df_not_treated):
#     np.ones(df_treated.shape[0])
#     propensity_treated = pscore.PropensityScore(df_treated,np.ones(df_treated.shape[0]))
#     treated_scores = propensity_treated.compute(method="Probit")
#     not_propensity_treated = pscore.PropensityScore(df_not_treated, np.zeros(df_not_treated.shape[0]))
#     not_treated_scores = not_propensity_treated.compute(method="Probit")
#     keep = np.where(0.05<not_treated_scores<0.95)
#     df_not_treated=df_not_treated.ix[keep]
#     keep = np.where(0.05 < treated_scores < 0.95)
#     df_treated=df_treated.ix[keep]
#     return df_treated,df_not_treated

