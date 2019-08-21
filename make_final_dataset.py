import pandas as pd
from preprocessing import read_local_csv
#----------------------------------------------------#
final_df = read_local_csv('', 'all_tweets_2018_cleaned.csv')
sentiment_df = read_local_csv('', '1.5_aggregate.csv')
location_df = read_local_csv('', '2_location_labelled_new.csv')
activity_df = read_local_csv('', '3_activity_labelled.csv')
time_df = read_local_csv('', '4_time_analysis.csv')

final_df['sentiment'] = sentiment_df['sentiment']
final_df['location_type'] = location_df['type']
final_df['activity'] = activity_df['activity']
final_df['working_day'] = time_df['working_day']
final_df['working_hours'] = time_df['working_hours']

final_df = final_df[final_df['country']=='Hong Kong']
final_df = final_df[final_df['lang']=='en']
final_df = final_df[final_df['location_type']!='no coordinate']
final_df = final_df[final_df['lat'].notnull()]
final_df.to_csv('final_dataset.csv')