import numpy as np
import pandas as pd
import re
# ultimately not used since we are using cleaned file
df = pd.read_csv("all_top_users_tweets.csv", low_memory=False, lineterminator="\n")
def remove_people_places_links(raw_text):
    result1 = re.sub(r'I\'m at [\S\s]+', '', raw_text)
    result2 = re.sub(r'https?://.*?', '', result1)
    result3 = re.sub(r'@.*? ', '', result2)
    final_result = re.sub(r'@_.*? ', '', result3)
    return final_result
# lambda is an anonymous function
df['text'] = df.apply(lambda row: remove_people_places_links(row['text']), axis=1)
df.to_csv('all_tweets_wo_people_places.csv')