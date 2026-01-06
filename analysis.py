# imports

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import and filter raw data
details_file = 'StormEvents_details_2025.csv'
details_input_df = pd.read_csv(details_file)
details_input_df = details_input_df[details_input_df['EVENT_TYPE'] == 'Tornado']

fatalities_file = 'StormEvents_fatalities_2025.csv'
fatalities_input_df = pd.read_csv(fatalities_file)
fatalities_input_df = fatalities_input_df[fatalities_input_df['FATALITY_LOCATION'] != 'Unknown']

# https://www.ncei.noaa.gov/stormevents/

# create new dataframes for cleaned data
detail_df = pd.DataFrame(None)
fatalities_df = pd.DataFrame(None)


#~~~~~~~      DETAILS CLEANING        ~~~~~
detail_df['event_id'] = details_input_df['EVENT_ID']
detail_df['fujita_scale'] = details_input_df['TOR_F_SCALE'].astype('str').str[2]
detail_df['state'] = details_input_df['STATE']
detail_df['state_fips'] = details_input_df['STATE_FIPS']
detail_df['zone'] = details_input_df['CZ_NAME']
detail_df['zone_fips'] = details_input_df['CZ_FIPS']
detail_df['start_month'] = details_input_df['BEGIN_YEARMONTH'].astype('str').str[4:6].astype('int')
detail_df['start_day'] = details_input_df['BEGIN_DAY']
detail_df['injuries'] = details_input_df['INJURIES_DIRECT'] + details_input_df['INJURIES_INDIRECT']
detail_df['deaths'] = details_input_df['DEATHS_DIRECT'] + details_input_df['DEATHS_INDIRECT']
detail_df.set_index('event_id', inplace = True)

# replacing EFU with -1 to show the intensity is unknown
detail_df['fujita_scale'] = detail_df['fujita_scale'].replace('U', -1)

#~~~~~~~      FATALITIES CLEANING        ~~~~~
fatalities_df['event_id'] = fatalities_input_df['EVENT_ID']
fatalities_df['fatality_location'] = fatalities_input_df['FATALITY_LOCATION']

# reformatting by number of fatalities by location and event
grouped_df = fatalities_df.groupby(['event_id', 'fatality_location']).size().reset_index(name='count')
pivot_df = grouped_df.pivot(index='event_id', columns='fatality_location', values='count').fillna(0)


#~~~~~~~      JOINING FATALITY LOCATIONS WITH TORNADO PHYSICAL CHARACTERISTICS        ~~~~~
joined_df = detail_df.join(pivot_df)

# replacing all NaN values from the join with 0s
location_cols = ['Ball Field', 'Boating', 'Business', 'Camping', 'Church', 'Golfing', 'Heavy Equipment/Construction', 'In Water', 'Mobile/Trailer Home', 'Other', 'Outside/Open Areas', 'Permanent Home', 'Permanent Structure', 'Under Tree', 'Vehicle/Towed Trailer']
joined_df[location_cols] = joined_df[location_cols].fillna(0)


#~~~~~~~      MACHINE LEARNING        ~~~~~

# predict how many casualties each tornado would cause using their physical characteristics
# ...and hence the level of emergency response required

# removing colunms which will not be used for training or testing
ml_df = detail_df.reset_index().drop(['event_id', 'state_fips', 'zone_fips', 'start_day'], axis=1)

# creating a human_damage metric which will feed into the ideal_response_level, giving different weightings to deaths vs injuries
ml_df['human_damage'] = (ml_df['deaths'] * 2) + ml_df['injuries']

# default value
ml_df['ideal_response_level'] =  'Nonexistent'

ml_df.loc[ml_df['human_damage'] >= 30, 'ideal_response_level'] = 'Very High'
ml_df.loc[(ml_df['human_damage'] >= 20) & (ml_df['human_damage'] < 30), 'ideal_response_level'] = 'High'
ml_df.loc[(ml_df['human_damage'] >= 10) & (ml_df['human_damage'] < 20), 'ideal_response_level'] = 'Medium'
ml_df.loc[(ml_df['human_damage'] >= 5) & (ml_df['human_damage'] < 10), 'ideal_response_level'] = 'Low'
ml_df.loc[(ml_df['human_damage'] >= 0) & (ml_df['human_damage'] < 5), 'ideal_response_level'] = 'Very Low'

ml_df.drop(['injuries', 'deaths', 'human_damage'], axis = 1, inplace = True)
print(ml_df)


# regression
# see if you can predict the number of tornadoes and other extreme weather events happening in the future