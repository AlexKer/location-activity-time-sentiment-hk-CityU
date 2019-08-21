import numpy as np
import pandas as pd
import googlemaps
import pprint
import json
import os, csv
import time
import re
#----------------------------------------------------#
def read_local_csv(path, filename):
    dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, dtype='str') 
    return dataframe
# Build new file to label location type
df = read_local_csv('', 'all_tweets_2018_cleaned.csv')
location_df = pd.DataFrame(columns=['user_id_str','type'])
# Common types mentioned in "I'm at x"
possible_types = ['MTR', 'Tunnel', 'Ferry', 'Bus', 'Airport', 'Store', 'Control Point', 'Restaurant', 'Mall', 'Tram'
                  'Park', 'Playground', 'Building', 'Immigration Port', 'School', 'Stadium', 'Fitness', 'Centre', 'Terminal']
# Define API Key and Client 
API_KEY = ''
gmaps = googlemaps.Client(key = API_KEY)
count = 0
for index, tweet in df.iterrows():
    #-------By Text------#
    # Delete URL and people tags 
    tweet['text'] = re.sub(r'https?://\S+', '', tweet['text'])
    tweet['text'] = re.sub(r'@\S+', '', tweet['text'])
    # Now obtain try to get place
    place = re.search(r'I\'m at [\S\s]+', tweet['text'])
    found = False
    # If user specifies location, see if type can be labelled
    if place is not None:
        # Group match object
        place = place.group()
        # Check in possible types list
        for type in possible_types:
            if type in place:
                place = type
                found = True
                location_df.loc[index] = [tweet['user_id_str'], place]
                break
    # Continue to next tweet if place can be labelled from looking at text
    if found:
        continue
    #------Google Places API------#
    coordinate = str(tweet['lat'])+','+str(tweet['lon'])
    # *Some tweets have no coorindates b/c of new cleaned file, just skip for now
    if coordinate == 'nan,nan':
        location_df.loc[index] = [tweet['user_id_str'], "no coordinate"]
        continue
    places_result = gmaps.places_nearby(location=coordinate, radius=10)
    time.sleep(2)
    print('Dealing with user {} that needs places API'.format(count))
    # Loop through each place in the results, and add all potential candidates
    stored_results = []
    num_places = 0
    for place in places_result['results']:
        # Place details call removed, since nearby request already returns types
        # store the results in a list object, ranked by prominence
        stored_results.append(place['types'])
        num_places = num_places + 1
        # consider max 5 places
        if num_places == 5:
            break
    print("inner loop finished")
    location_df.loc[index] = [tweet['user_id_str'], " ".join(str(x) for x in stored_results)]
    count = count + 1
    # # Comment out later
    # if count == 20:
    #     break
location_df.to_csv('location_labelled_new.csv')