import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from preprocessing import read_local_csv
import pandas as pd
import matplotlib.pyplot as plt
#----------------------------------------------------#
df = read_local_csv('dataset/', 'stats.csv')
print('Regression analysis starts..... ')
# No need for normalization everything is 0 or 1 
# Check the correlation matrix of independent variables and compute VIF value for each independent variable
# dataframe_for_correlation_matrix = df[df.columns.difference(['sentiment'])]
# draw_correlation_plot(dataframe_for_correlation_matrix)
# result_vif_series = compute_vif(dataframe_for_correlation_matrix)
# print(result_vif_series)
df = df.apply(pd.to_numeric)
# y = df['sentiment']
# X = df.drop(columns=['sentiment'])
formula_location_and_time = 'sentiment ~ C(bakery)+C(cafe)+C(doctor)+C(gym)+C(hair_care)+C(hospital)\
        +C(laundry)+C(lodging)+C(meal_delivery)+C(meal_takeaway)+C(pharmacy)+C(physiotherapist)+C(supermarket)\
        +C(Fitness)+C(library)+C(school)+C(School)+C(atm)+C(car_dealer)+C(car_rental)+C(car_repair)+C(car_wash)\
        +C(cemetery)+C(church)+C(city_hall)+C(convenience_store)+C(courthouse)+C(dentist)+C(electrician)+C(embassy)\
        +C(fire_station)+C(funeral_home)+C(funiture_store)+C(gas_station)+C(hardware_store)+C(hindu_temple)\
        +C(home_goods_store)+C(insurance_agency)+C(lawyer)+C(local_government)+C(locksmith)+C(mosque)+C(moving_company)\
        +C(plumber)+C(police)+C(post_office)+C(real_estate_agency)+C(roofing_contractor)+C(rv_park)+C(synagogue)\
        +C(veterinary_care)+C(Building)+C(Centrepolitical)+C(amusement_park)+C(aquarium)+C(art_gallery)+C(beauty_salon)\
        +C(bicycle_store)+C(book_store)+C(bowling_alley)+C(campground)+C(casino)+C(clothing_store)+C(electronics_store)\
        +C(florist)+C(jewelry_store)+C(liquor_store)+C(movie_rental)+C(movie_theater)+C(museum)+C(night_club)+C(painter)\
        +C(park)+C(pet_store)+C(restaurant)+C(shoe_store)+C(shopping_mall)+C(spa)+C(stadium)+C(store)+C(travel_agency)\
        +C(zoo)+C(Store)+C(Restaurant)+C(Mall)+C(Park)+C(Playground)+C(Stadium)+C(airport)+C(bus_station)+C(subway_station)\
        +C(taxi_stand)+C(train_station)+C(transit_station)+C(route)+C(MTR)+C(Tunnel)+C(Ferry)+C(Bus)+C(Airport)\
        +C(Control_Point)+C(Tram)+C(Immigration_Port)+C(Terminal)+C(working_day)+C(working_hours)'
# Needs to avoid dummy variable trap (multicolinearity): exclude one as baseline - Travel excluded if we use activity_list
# Remember there is default dummy variable methods, but we already one-hot-encoded in 'prepare_for_stats.py'
formula_activity_and_time = 'sentiment ~ C(education_training)+C(basic_necessities)\
        +C(unpaid_activities)+C(free_time_leisure_activities)+C(working_day)+C(working_hours)'
#Run two OLS regression, one with all location types and time, one with activity and time
OLSregression = smf.ols(formula=formula_location_and_time, data=df).fit()
print("First OLS Model:")
print(OLSregression.summary())
OLSregression = smf.ols(formula=formula_activity_and_time, data=df).fit()
print("Second OLS Model:")
print(OLSregression.summary())

# MNlogit
activity_and_time_col = ['education_training','basic_necessities','unpaid_activities',
    'free_time_leisure_activities','working_day','working_hours']
X = df.drop(columns=['sentiment','education_training','basic_necessities','unpaid_activities','free_time_leisure_activities','travel'])
location_and_time_col = list(X.columns.values)
# location_and_time col fails to converge, so we only look at activity_time_col
MNlogit = sm.MNLogit(df['sentiment'], df[activity_and_time_col ]).fit()
print("Multinominal Logisit Regression:")
print(MNlogit.summary())