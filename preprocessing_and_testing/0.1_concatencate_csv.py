import os
import glob
import pandas as pd
# cwd = os.getcwd()
# print(cwd)
os.chdir("/Users/alexker/Desktop/CityU/raw_tweets")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f,encoding='latin-1') for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)