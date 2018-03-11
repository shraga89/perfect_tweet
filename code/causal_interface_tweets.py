import pandas as pd

import preprocess_tweets as p
from average_treatement_effect_tweets import get_average_treatment_effect

print("in preprocess")
df = pd.read_csv("tweets_updated.csv",encoding = "ISO-8859-1")
df = p.add_extra_attributes(df)
df=p.add_treatment_flag_for_dots(df, (".", "!", "?"))
# df=p.add_treatment_flag_for_image(df)
df = df[df.popularity!=-1]
treatment_popularity, no_treatment_popularity,treatment_features,no_treatment_features = p.get_averages_for_tests(df,"contains_dot")
# treatment_popularity, no_treatment_popularity,treatment_features,no_treatment_features = p.get_averages_for_tests(df,"images")

ATE = get_average_treatment_effect(treatment_features,no_treatment_features,no_treatment_popularity,treatment_popularity ,5)
print(ATE)




