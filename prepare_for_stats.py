import pandas as pd
from preprocessing import read_local_csv
import re
#----------------------------------------------------#
df = read_local_csv('dataset/', 'final_dataset.csv')
location_type_list = [
    'bakery','cafe','doctor','gym','hair_care','hospital','laundry','lodging','meal_delivery',
    'meal_takeaway','pharmacy','physiotherapist','supermarket','Fitness',
    'library','school','School',
    'atm','car_dealer','car_rental','car_repair','car_wash','cemetery','church','city_hall',
    'convenience_store','courthouse','dentist','electrician','embassy','fire_station','funeral_home',
    'funiture_store','gas_station','hardware_store','hindu_temple','home_goods_store','insurance_agency',
    'lawyer','local_government','locksmith','mosque','moving_company','plumber','police','post_office',
    'real_estate_agency','roofing_contractor','rv_park','synagogue','veterinary_care','Building','Centre'
    'political',
    'amusement_park','aquarium','art_gallery','beauty_salon','bicycle_store','book_store','bowling_alley',
    'campground','casino','clothing_store','electronics_store','florist','jewelry_store','liquor_store',
    'movie_rental','movie_theater','museum','night_club','painter','park','pet_store','restaurant',
    'shoe_store','shopping_mall','spa','stadium','store','travel_agency','zoo','Store','Restaurant','Mall',
    'Park','Playground','Stadium',
    'airport','bus_station','subway_station','taxi_stand','train_station','transit_station','route',
    'MTR','Tunnel','Ferry','Bus','Airport','Control Point','Tram','Immigration Port','Terminal'
    ]
activity_list = ['education_training', 'basic_necessities', 'unpaid_activities', 'free_time_leisure_activities', 'travel']
time_of_day = ['working_day', 'working_hours']
# If time permits, will add district and personalities back
independent_variables = location_type_list+activity_list+time_of_day
dependent_variable = ['sentiment']
stats_df = pd.DataFrame(columns=dependent_variable+independent_variables)
# Reformat the current location_type column we have
def reformat_location_type(type_string):
    type_string = re.sub('[\[\]\'\,]','', type_string)
    type_list = type_string.split()
    return type_list
type_list = [reformat_location_type(i) for i in df['location_type']]
df['location_type'] = type_list
# Mark dummy variables, in other words apply one hot encoding: 1 and 0
def transform_to_dummy(row, cnt):
    stats_df.loc[cnt] = None
    for type in location_type_list:
        if type in row['location_type']:
            stats_df.loc[cnt][type] = 1
        else:
            stats_df.loc[cnt][type] = 0
    for activity in activity_list:
        if activity == row['activity']:
            stats_df.loc[cnt][activity] = 1
        else:
            stats_df.loc[cnt][activity] = 0
    for i in time_of_day:
        if str(row[i]) == 'True':
            stats_df.loc[cnt][i] = 1
        else:
            stats_df.loc[cnt][i] = 0
cnt = 0
for _, row in df.iterrows():
    transform_to_dummy(row, cnt) 
    cnt += 1
# manually paste sentiment column in because index error
stats_df.to_csv('stats.csv')