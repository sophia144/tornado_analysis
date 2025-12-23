#imports

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#import and filter raw data
details_file = 'StormEvents_details_2025.csv'
details_input_df = pd.read_csv(details_file)
details_input_df = details_input_df[details_input_df['EVENT_TYPE'] == 'Tornado']

fatalities_file = 'StormEvents_fatalities_2025.csv'
fatalities_input_df = pd.read_csv(fatalities_file)
fatalities_input_df = fatalities_input_df[fatalities_input_df['FATALITY_LOCATION'] != 'Unknown']

# https://www.ncei.noaa.gov/stormevents/

#create new dataframes for cleaned data
detail_df = pd.DataFrame(None)
fatalities_df = pd.DataFrame(None)

#~~~~~~~      DETAILS CLEANING        ~~~~~
detail_df['event_id'] = details_input_df['EVENT_ID']
detail_df['state'] = details_input_df['STATE']
detail_df['state_fips'] = details_input_df['STATE_FIPS']
detail_df['zone'] = details_input_df['CZ_NAME']
detail_df['zone_fips'] = details_input_df['CZ_FIPS']
detail_df['start_year'] = details_input_df['BEGIN_YEARMONTH'].astype('str').str[0:4]
detail_df['start_month'] = details_input_df['BEGIN_YEARMONTH'].astype('str').str[4:6]
detail_df['start_day'] = details_input_df['BEGIN_DAY']
detail_df['injuries'] = details_input_df['INJURIES_DIRECT'] + details_input_df['INJURIES_INDIRECT']
detail_df['deaths'] = details_input_df['DEATHS_DIRECT'] + details_input_df['DEATHS_INDIRECT']
detail_df['fujita_scale'] = details_input_df['TOR_F_SCALE']

#~~~~~~~      FATALITIES CLEANING        ~~~~~
fatalities_df['event_id'] = fatalities_input_df['EVENT_ID']
fatalities_df['fatality_location'] = fatalities_input_df['FATALITY_LOCATION']

print(fatalities_df)

# machine learning
# predict how many casualties each tornado would cause
# ...and hence the level of emergency response required

# regression
# see if you can predict the number of tornadoes and other extreme weather events happening in the future