import pandas as pd
from average_treatement_effect_users import get_average_treatment_effect
import preprocess_users as p
print("in preprocess")
df = pd.read_csv("tweets_updated.csv",encoding = "ISO-8859-1")
df = p.add_extra_attributes(df)
df=p.add_treatment_flag_for_dots(df, (".", "!", "?"))
# df=p.add_treatment_flag_for_image(df)
df = df[df.popularity!=-1]
keys = ["user.id","user.listed_count", "user.followers_count", "user.friends_count", "user.statuses_count","user.favourites_count"]
treatment_popularity,no_treatment_popularity,treatment_features,no_treatment_features = p.get_averages_for_tests_dots(df, keys)
# treatment_popularity,no_treatment_popularity,treatment_features,no_treatment_features = p.get_averages_for_tests_images(df, keys)

ATE = get_average_treatment_effect(treatment_features,no_treatment_features,no_treatment_popularity,treatment_popularity,5)
print(ATE)




