# imports
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np





# https://www.ncei.noaa.gov/stormevents/

# import and filter raw data
# details
details_file_2024 = 'StormEvents_details_2024.csv'
details_file_2025 = 'StormEvents_details_2025.csv'
details_df_2024 = pd.read_csv(details_file_2024)
details_df_2025 = pd.read_csv(details_file_2025)
details_input_df = pd.concat([details_df_2024, details_df_2025], ignore_index = True)
# creating a duplicate table for analysis needing all records
all_weather_input_df = details_input_df
# filtering out all weather events which are not tornadoes
details_input_df = details_input_df[details_input_df['EVENT_TYPE'] == 'Tornado']

# fatalities
fatalities_file_2024 = 'StormEvents_fatalities_2024.csv'
fatalities_file_2025 = 'StormEvents_fatalities_2025.csv'
fatalities_df_2024 = pd.read_csv(fatalities_file_2024)
fatalities_df_2025 = pd.read_csv(fatalities_file_2025)
fatalities_input_df = pd.concat([fatalities_df_2024, fatalities_df_2025], ignore_index = True)

# locations
locations_file_2024 = 'StormEvents_locations_2024.csv'
locations_file_2025 = 'StormEvents_locations_2025.csv'
locations_df_2024 = pd.read_csv(locations_file_2024)
locations_df_2025 = pd.read_csv(locations_file_2025)
locations_input_df = pd.concat([locations_df_2024, locations_df_2025], ignore_index = True)

# create new dataframes for cleaned data
detail_df = pd.DataFrame(None)
all_weather_df = pd.DataFrame(None)
fatalities_df = pd.DataFrame(None)
locations_df = pd.DataFrame(None)








#~~~~~~~      DETAILS CLEANING        ~~~~~
detail_df['episode_id'] = details_input_df['EPISODE_ID']
detail_df['event_id'] = details_input_df['EVENT_ID']
detail_df['fujita_scale'] = details_input_df['TOR_F_SCALE'].astype('str').str[2]
detail_df['state'] = details_input_df['STATE']
detail_df['state_fips'] = details_input_df['STATE_FIPS']
detail_df['zone'] = details_input_df['CZ_NAME']
detail_df['zone_fips'] = details_input_df['CZ_FIPS']
detail_df['start_year'] = details_input_df['BEGIN_YEARMONTH'].astype('str').str[0:4].astype('int')
detail_df['start_month'] = details_input_df['BEGIN_YEARMONTH'].astype('str').str[4:6].astype('int')
detail_df['start_day'] = details_input_df['BEGIN_DAY']
detail_df['injuries'] = details_input_df['INJURIES_DIRECT'] + details_input_df['INJURIES_INDIRECT']
detail_df['deaths'] = details_input_df['DEATHS_DIRECT'] + details_input_df['DEATHS_INDIRECT']
# replacing EFU with -1 to show the intensity is unknown
detail_df['fujita_scale'] = detail_df['fujita_scale'].replace('U', -1)


#~~~~~~~      ALL WEATHER CLEANING        ~~~~~
all_weather_df['episode_id'] = all_weather_input_df['EPISODE_ID']
all_weather_df['event_id'] = all_weather_input_df['EVENT_ID']
all_weather_df['event_type'] = all_weather_input_df['EVENT_TYPE']
all_weather_df['state'] = all_weather_input_df['STATE']
all_weather_df['state_fips'] = all_weather_input_df['STATE_FIPS']
all_weather_df['zone'] = all_weather_input_df['CZ_NAME']
all_weather_df['zone_fips'] = all_weather_input_df['CZ_FIPS']
all_weather_df['start_year'] = all_weather_input_df['BEGIN_YEARMONTH'].astype('str').str[0:4].astype('int')
all_weather_df['start_month'] = all_weather_input_df['BEGIN_YEARMONTH'].astype('str').str[4:6].astype('int')
all_weather_df['start_day'] = all_weather_input_df['BEGIN_DAY']
all_weather_df['injuries'] = all_weather_input_df['INJURIES_DIRECT'] + all_weather_input_df['INJURIES_INDIRECT']
all_weather_df['deaths'] = all_weather_input_df['DEATHS_DIRECT'] + all_weather_input_df['DEATHS_INDIRECT']



#~~~~~~~      FATALITIES CLEANING        ~~~~~
fatalities_df['event_id'] = fatalities_input_df['EVENT_ID']
fatalities_df['fatality_location'] = fatalities_input_df['FATALITY_LOCATION']
# reformatting by number of fatalities by location and event
grouped_df = fatalities_df.groupby(['event_id', 'fatality_location']).size().reset_index(name='count')
pivot_df = grouped_df.pivot(index='event_id', columns='fatality_location', values='count').fillna(0)
# fixing datatypes
pivot_df = pivot_df.astype('Int64')
pivot_df.index = pivot_df.index.astype('Int64')



#~~~~~~~      LOCATIONS CLEANING        ~~~~~
locations_df['episode_id'] = locations_input_df['EPISODE_ID']
locations_df['event_id'] = locations_input_df['EVENT_ID']
locations_df['location'] = locations_input_df['LOCATION']
locations_df['direction'] = locations_input_df['AZIMUTH']
locations_df['range'] = locations_input_df['RANGE']

# joining locations to details
detail_df.set_index('event_id', inplace = True)
locations_df.set_index('event_id', inplace = True)
locations_df.drop(['episode_id'], axis = 1, inplace = True)
location_detail_df = detail_df.join(locations_df)

# forced to remove all events without a corresponding Location for the model, as it will require this data
# attempted to research why some rows do not have corresponding Location entries, but have not found why
# this leaves me with 3118 rows
location_detail_df = location_detail_df.dropna(how='any')









#~~~~~~~      JOINING FATALITY LOCATIONS WITH TORNADO PHYSICAL CHARACTERISTICS        ~~~~~

# this will be used to analyse in which situations the highest numbers of fatalities occur in each state

joined_df = detail_df.join(pivot_df)

location_cols = ['Unknown', 'Ball Field', 'Boating', 'Business', 'Camping', 'Church', 'Golfing', 'Heavy Equipment/Construction', 'In Water', 'Mobile/Trailer Home', 'Other', 'Outside/Open Areas', 'Permanent Home', 'Permanent Structure', 'Under Tree', 'Vehicle/Towed Trailer']
joined_df[location_cols] = joined_df[location_cols].fillna(0)






#~~~~~~~      INITIAL ANALYSIS 1        ~~~~~

# what proportion of extreme weather events in the US do tornadoes make up?

all_weather_count = len(all_weather_df.index)
tornado_count = len(all_weather_df[all_weather_df['event_type'] == 'Tornado'].index)
tornado_occurrence_proportion = tornado_count/all_weather_count
# tornadoes made up 2.83% of extreme weather events in 2024-5

# deciding when to start bucketing weather events into an 'other' column
weather_by_occurrence = all_weather_df['event_type'].value_counts()
position_10 = weather_by_occurrence.nlargest(10).iloc[-1]

bucketing_threshold = position_10

weather_by_occurrence = weather_by_occurrence.to_frame(name = 'occurrences')
weather_by_occurrence['bucket'] = weather_by_occurrence.index
# where the number of occurrences is below the bucketing threshold, assign it to the bucket Other
mask = weather_by_occurrence['occurrences'] < bucketing_threshold
weather_by_occurrence.loc[mask, 'bucket'] = 'Other'

# grouping by bucket
weather_by_occurrence = weather_by_occurrence.groupby('bucket')['occurrences'].sum()
weather_by_occurrence.sort_values(ascending=False, inplace=True)


# creating the bar graph

# setting up parameters for plotting
x_axis = weather_by_occurrence.index
x_axis_label = "Event"
y_axis = weather_by_occurrence.values
y_axis_label = "Occurrences"
title = "Count of Weather Events by Type across the USA (2024-5)"


# plotting and formatting
plt.figure(figsize=(8, 10))          
x_pos = np.arange(len(x_axis))
plt.bar(x_pos, y_axis, align='center')      
plt.xticks(x_pos, x_axis, rotation=45, ha='right') 
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.tight_layout()                    

plt.show()






#~~~~~~~      INITIAL ANALYSIS 2        ~~~~~

# what proportion of weather related injuries and fatalities in the US do tornadoes make up?

damage_grouping = all_weather_df.groupby('event_type')[['injuries', 'deaths']].sum()

all_weather_injuries = damage_grouping['injuries'].sum()
all_weather_deaths = damage_grouping['deaths'].sum()
tornado_injuries = int(damage_grouping[damage_grouping.index == 'Tornado']['injuries'].iloc[0])
tornado_deaths = int(damage_grouping[damage_grouping.index == 'Tornado']['deaths'].iloc[0])

tornado_injury_proportion = tornado_injuries/all_weather_injuries
# tornadoes were responsible for 31.81% of weather related injuries in 2024-5
# 6 times overrepresented in casualty data
tornado_fatality_proportion = tornado_deaths/all_weather_deaths
# tornadoes were responsible for 6.54% of weather related injuries in 2024-5

def output_context_statistics(tornado_occurrence_proportion, tornado_injury_proportion, tornado_fatility_proportion):
    print('Occurrence Proportion: ', tornado_occurrence_proportion)
    print('Injury Proportion: ', tornado_injury_proportion)
    print('Fatality Proportion: ', tornado_fatality_proportion)



# graphing
# deciding when to start bucketing weather events into an 'other' column
position_10 = damage_grouping.nlargest(10, 'injuries').iloc[-1]
bucketing_threshold = position_10['injuries']
damage_grouping['bucket'] = damage_grouping.index

# where the number of occurrences is below the bucketing threshold, assign it to the bucket Other
mask = damage_grouping['injuries'] < bucketing_threshold
damage_grouping.loc[mask, 'bucket'] = 'Other'

# grouping by bucket
damage_grouping = damage_grouping.groupby('bucket')[['injuries', 'deaths']].sum()
damage_grouping.sort_values(ascending=False, inplace=True, by='injuries')

# creating the bar graph

# setting up parameters for plotting
x_axis = damage_grouping.index
x_axis_label = "Event"
y_axis_1 = damage_grouping['injuries']
y_axis_2 = damage_grouping['deaths']
y_axis_label = "Casualties"
title = "Casualties by Type of Weather Event across the USA (2024-5)"


# plotting and formatting
plt.figure(figsize=(8, 10))          
x_pos = np.arange(len(x_axis))
width = 0.4
plt.bar(x_pos - width/2, y_axis_1, width=width, align='center', label='Injuries')    
plt.bar(x_pos + width/2, y_axis_2, width=width, align='center', label='Deaths')   
plt.xticks(x_pos, x_axis, rotation=45, ha='right') 
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.tight_layout()                    
plt.legend()

plt.show()






#~~~~~~~      INITIAL ANALYSIS 3        ~~~~~

# which areas of the US are worst affected by tornadoes?


# by injuries and deaths
state_grouping = detail_df.groupby('state')[['injuries', 'deaths']].sum()

total_tornado_injuries = state_grouping['injuries'].sum()
total_tornado_deaths = state_grouping['deaths'].sum()

most_injuries = state_grouping.loc[state_grouping['injuries'].idxmax()]
most_deaths = state_grouping.loc[state_grouping['injuries'].idxmax()]


def output_affected_states(most_injuries, most_deaths):
    injuries_state = most_injuries.name.title()
    injuries_count = most_injuries['injuries']
    print('The worst-affected state by injuries is', injuries_state, 'with a total injury count of', injuries_count)
    deaths_state = most_deaths.name.title()
    deaths_count = most_deaths['deaths']
    print('The worst-affected state by death toll is', deaths_state, 'with a total fatality count of', deaths_count)


# graphing
# deciding when to start bucketing s into an 'other' column
position_10 = state_grouping.nlargest(10, 'injuries').iloc[-1]
bucketing_threshold = position_10['injuries']
state_grouping['bucket'] = state_grouping.index

# where the number of occurrences is below the bucketing threshold, assign it to the bucket Other
mask = state_grouping['injuries'] < bucketing_threshold
state_grouping.loc[mask, 'bucket'] = 'Other'
state_grouping = state_grouping[state_grouping['bucket'] != 'Other']
top_10_states = list(state_grouping.index)

# grouping by bucket
state_grouping = state_grouping.groupby('bucket')[['injuries', 'deaths']].sum()
state_grouping.sort_values(ascending=False, inplace=True, by='injuries')



# UNWEIGHTED CASUALTIES BY STATE
# creating the bar graph

# setting up parameters for plotting
x_axis = state_grouping.index
x_axis_label = "State"
y_axis_1 = state_grouping['injuries']
y_axis_2 = state_grouping['deaths']
y_axis_label = "Casualties"
title = "Tornado Casualties by State across the USA (2024-5)"


# plotting and formatting
plt.figure(figsize=(8, 10))          
x_pos = np.arange(len(x_axis))
width = 0.4
plt.bar(x_pos - width/2, y_axis_1, width=width, align='center', label='Injuries')    
plt.bar(x_pos + width/2, y_axis_2, width=width, align='center', label='Deaths')   
plt.xticks(x_pos, x_axis, rotation=45, ha='right') 
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.tight_layout()                    
plt.legend()

plt.show()


# calculating weighted casualties (per tornado event)
occurrences_by_state = detail_df.value_counts('state')
occurrences_by_state = occurrences_by_state[occurrences_by_state.index.isin(top_10_states)]

weighted_data = state_grouping.join(occurrences_by_state)
weighted_data['injuries_per_event'] = weighted_data['injuries']/weighted_data['count']
weighted_data['deaths_per_event'] = weighted_data['deaths']/weighted_data['count']




# WEIGHTED CASUALTIES BY STATE - ie. where are tornadoes the deadliest
# creating the bar graph

# setting up parameters for plotting
x_axis = weighted_data.index
x_axis_label = "State"
y_axis_1 = weighted_data['injuries_per_event']
y_axis_2 = weighted_data['deaths_per_event']
y_axis_label = "Casualties per Tornado Event"
title = "Average Casualties per Tornado Event by State across the USA (2024-5)"


# plotting and formatting
plt.figure(figsize=(8, 10))          
x_pos = np.arange(len(x_axis))
width = 0.4
plt.bar(x_pos - width/2, y_axis_1, width=width, align='center', label='Injuries per Tornado Event')    
plt.bar(x_pos + width/2, y_axis_2, width=width, align='center', label='Deaths per Tornado Event')   
plt.xticks(x_pos, x_axis, rotation=45, ha='right') 
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.tight_layout()                    
plt.legend()

plt.show()

exit()

#~~~~~~~      INITIAL ANALYSIS 4        ~~~~~

# how do most casualties occur, and does this vary by state?

print(detail_df)


#~~~~~~~      INITIAL ANALYSIS 5        ~~~~~

# are tornadoes of different intensities evenly spread between states?



#~~~~~~~      MACHINE LEARNING        ~~~~~

# predict how many casualties each tornado would cause using their physical characteristics
# ...and hence the level of emergency response required

# removing colunms which will not be used for training or testing
ml_df = location_detail_df.reset_index().drop(['event_id', 'state_fips', 'zone_fips', 'start_day', 'location'], axis=1)

# creating a human_damage metric which will feed into the ideal_response_level, giving different weightings to deaths vs injuries
ml_df['human_damage'] = (ml_df['deaths'] * 4) + ml_df['injuries']

# default value for response metric
ml_df['ideal_response_level'] =  'Low'
ml_df.loc[(ml_df['human_damage'] >= 1) & (ml_df['human_damage'] <= 6), 'ideal_response_level'] = 'Medium'
ml_df.loc[(ml_df['human_damage'] >= 7), 'ideal_response_level'] = 'High'

# figuring out where class boundaries should be
# print(ml_df['ideal_response_level'].value_counts())

# dropping the columns used to calculate the classification metric
ml_df.drop(['injuries', 'deaths', 'human_damage'], axis = 1, inplace = True)

x = ml_df.drop(['ideal_response_level'], axis = 1)
y = np.array(ml_df['ideal_response_level'])

# encoding
x = pd.get_dummies(x)

le = LabelEncoder()
y = le.fit_transform(y) 

# splitting data into training and testing populations
# 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y) 
clf = DecisionTreeClassifier(class_weight='balanced', random_state=13) # remember to add max_depth later
clf = clf.fit(x_train, y_train)
# predict the response for test dataset
y_pred = clf.predict(x_test)
y_probs = np.array(clf.predict_proba(x_test))[:, 1]

# cross-validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, x, y, cv = k_folds)

def return_stats_1(scores):
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))

# calculates and prints the precision, recall, and F1 score for each class
def return_stats_2(y_test, y_pred, class_names=le.classes_):
    print(list(class_names))

    precision = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    recall = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()

    print("Precision (Per Class):", precision)
    print("Recall (Per Class):", recall)
    print("F1 Score (Per Class):", f1)

# return_stats_1(scores)
# print('\n')
# return_stats_2(y_test, y_pred)






#~~~~~~~      REGRESSION        ~~~~~

# see if you can predict the number of tornadoes and other extreme weather events happening in the future
regression_df = detail_df

# converting the three integer columns into one date column
regression_df['date'] = pd.to_datetime(
    dict(year = regression_df['start_year'], month = regression_df['start_month'], day = regression_df['start_day']),
    errors = 'raise'
)
# creating a daily cumulative measure
date_count = regression_df['date'].value_counts().sort_index()
cumulative_count = date_count.cumsum()

# creating a weekly standalone measure
weekly_count = date_count.resample('W').sum()


# ~~~~~~~~~~~~~~~~~~~~~~ GRAPH 1 ~~~~~~~~~~~~~~~~~~~~~~

plt.figure()

# setting up parameters for plotting
x_axis = cumulative_count.index
x_axis_label = "Date"
y_axis = cumulative_count.values
y_axis_label = "Cumulative Tornado Count"

title = "Cumulative Count of Tornadoes across the USA"

# plotting the main data
plt.grid(alpha=0.5)
plt.scatter(x_axis, y_axis, s=8, label='Raw Data')
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.xticks(rotation=45)

# setting up containers for stats
order_vals = []
chi_squared_vals = []
chi_squared_dof_vals = []
bic_vals = []
uncertainty = 0.1

# computing statistics for each polynomial
for order in range(1, 8):
    # changing the x_axis datatypes from dates to integers as polyfit only works on numbers
    x_num = mdates.date2num(x_axis)
    # centering the axis around the mean to avoid large numbers
    x_mean = x_num.mean()
    x_num = x_num - x_mean

    order_vals.append(order)

    #calculations and plotting
    coefficients = np.polyfit(x_num, y_axis, order)
    poly_function = np.poly1d(coefficients)

    #chi squared calculation
    residuals = poly_function(x_num) - y_axis
    chi_squared = 0
    for residual in residuals:
        chi_squared += (residual ** 2) / (uncertainty ** 2)
    chi_squared_vals.append(chi_squared)

    #chi squared by degrees of freedom
    degrees_of_freedom = len(x_axis) - (order + 1)
    chi_squared_dof = chi_squared/degrees_of_freedom
    chi_squared_dof_vals.append(chi_squared_dof)

    #bic calculations
    bic = chi_squared + ((order + 1) * np.log(len(x_axis)))
    bic_vals.append(bic)


# finding the best fit

best_bic = bic_vals.index(min(bic_vals)) + 1
coefficients = np.polyfit(x_num, y_axis, best_bic)
poly_function = np.poly1d(coefficients)
plt.plot(x_axis, poly_function(x_num), color='orange', alpha=0.8, lw=1.8, label='Polynomial Fit')

plt.legend()
plt.show()






# ~~~~~~~~~~~~~~~~~~~~~~ GRAPH 2 ~~~~~~~~~~~~~~~~~~~~~~

plt.figure()

# setting up parameters for plotting
x_axis = weekly_count.index
x_axis_label = "Week Beginning"
y_axis = weekly_count.values
y_axis_label = "Weekly Tornado Count"

title = "Weekly Count of Tornadoes across the USA"

# plotting the main data
plt.grid(alpha=0.5)
plt.plot(x_axis, y_axis, label='Raw Data')
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(title, pad=20)
plt.xticks(rotation=45)

# setting up containers for stats
order_vals = []
chi_squared_vals = []
chi_squared_dof_vals = []
bic_vals = []
uncertainty = 0.1

for order in range(1, 8):
    # changing the x_axis datatypes from dates to integers as polyfit only works on numbers
    x_num = mdates.date2num(x_axis)
    # centering the axis around the mean to avoid large numbers
    x_mean = x_num.mean()
    x_num = x_num - x_mean

    order_vals.append(order)

    #calculations and plotting
    coefficients = np.polyfit(x_num, y_axis, order)
    poly_function = np.poly1d(coefficients)

    #chi squared calculation
    residuals = poly_function(x_num) - y_axis
    chi_squared = 0
    for residual in residuals:
        chi_squared += (residual ** 2) / (uncertainty ** 2)
    chi_squared_vals.append(chi_squared)

    #chi squared by degrees of freedom
    degrees_of_freedom = len(x_axis) - (order + 1)
    chi_squared_dof = chi_squared/degrees_of_freedom
    chi_squared_dof_vals.append(chi_squared_dof)

    #bic calculations
    bic = chi_squared + ((order + 1) * np.log(len(x_axis)))
    bic_vals.append(bic)

# plotting the best fit
best_bic = bic_vals.index(min(bic_vals)) + 1
coefficients = np.polyfit(x_num, y_axis, best_bic)
poly_function = np.poly1d(coefficients)
plt.plot(x_axis, poly_function(x_num), color='orange', alpha=0.7, lw=1.8, label='Polynomial Fit')

plt.legend()
plt.show()