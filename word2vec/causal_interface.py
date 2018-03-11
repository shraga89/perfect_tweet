import pandas as pd
from word2vec_platform.average_treatement_effect import get_average_treatment_effect
import word2vec_platform.preprocess as p
print("in preprocess")
df = pd.read_csv("../tweets_updated.csv",encoding = "ISO-8859-1")
df = p.add_extra_attributes(df)
df=p.add_treatment_flag_for_dots(df, (".", "!", "?"))
df = df[df.popularity!=-1]
treatment_features,no_treatment_features = p.get_averages_for_tests_dots(df)

ATE = get_average_treatment_effect(treatment_features,no_treatment_features)
print(ATE)




