import pandas as pd
import re
from preprocessing import read_local_csv
#----------------------------------------------------#
location_df = read_local_csv('', 'location_labelled_new.csv')
# Activity labelling according to 2015 Thematic Household Survey Report by Census and Statistics Dpt HK
# * activities outside HK category is removed since all tweets are within bounding box
# * paid activities removed since we cannot determine the jobs of users
activity_dict = {}
activity_dict.update(dict.fromkeys(
    ['bakery','cafe','doctor','gym','hair_care','hospital','laundry','lodging','meal_delivery',
    'meal_takeaway','pharmacy','physiotherapist','supermarket','Fitness'
    ], 'basic_necessities'))
activity_dict.update(dict.fromkeys(['library','school','School'], 'education_training'))
activity_dict.update(dict.fromkeys(
    ['atm','car_dealer','car_rental','car_repair','car_wash','cemetery','church','city_hall',
    'convenience_store','courthouse','dentist','electrician','embassy','fire_station','funeral_home',
    'funiture_store','gas_station','hardware_store','hindu_temple','home_goods_store','insurance_agency',
    'lawyer','local_government','locksmith','mosque','moving_company','plumber','police','post_office',
    'real_estate_agency','roofing_contractor','rv_park','synagogue','veterinary_care','Building','Centre'
    'political'
    ], 'unpaid_activities')) # mundane and tedious tasks that no one does for fun
activity_dict.update(dict.fromkeys(
    ['amusement_park','aquarium','art_gallery','beauty_salon','bicycle_store','book_store','bowling_alley',
    'campground','casino','clothing_store','electronics_store','florist','jewelry_store','liquor_store',
    'movie_rental','movie_theater','museum','night_club','painter','park','pet_store','restaurant',
    'shoe_store','shopping_mall','spa','stadium','store','travel_agency','zoo','Store','Restaurant','Mall',
    'Park','Playground','Stadium'
    ], 'free_time_leisure_activities'))
activity_dict.update(dict.fromkeys(
    ['airport','bus_station','subway_station','taxi_stand','train_station','transit_station','route'
    'MTR','Tunnel','Ferry','Bus','Airport','Control Point','Tram','Immigration Port','Terminal'
    ], 'travel'))
# Using possible_types list and Google Places types, determine the most likely activity
def get_activity(type_string):
    """ return most likely activity based on frequency """
    type_string = re.sub('[\[\]\'\,]','', type_string)
    type_list = type_string.split()
    activity_cnt_dict = {
        'basic_necessities': 0, 
        'education_training': 0, 
        'unpaid_activities': 0,
        'free_time_leisure_activities': 0,
        'travel': 0
        }
    for type in type_list:
        if type in activity_dict:
            activity = activity_dict[type]
            activity_cnt_dict[activity] += 1
    most_likely_activity = max(activity_cnt_dict , key=activity_cnt_dict.get)
    return most_likely_activity
activity_df = pd.DataFrame(columns=['user_id_str','activity'])
activity_df['user_id_str'] = location_df['user_id_str'] 
activity_df['activity'] = [get_activity(str(type)) for type in location_df['type']]
activity_df.to_csv('activity_labelled.csv')