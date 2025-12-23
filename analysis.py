import pandas as pd
import numpy as np

#import and filter raw data
details_file = 'StormEvents_details_2025.csv'
input_df = pd.read_csv(details_file)
input_df = input_df[input_df['EVENT_TYPE'] == 'Tornado']

# https://www.ncei.noaa.gov/stormevents/

#create new dataframe for cleaned data
detail_df = pd.DataFrame(None)

#~~~~~~~      CLEANING        ~~~~~
detail_df['state'] = input_df['STATE']
detail_df['state_fips'] = input_df['STATE_FIPS']
detail_df['zone'] = input_df['CZ_NAME']
detail_df['zone_fips'] = input_df['CZ_FIPS']
detail_df['start_year'] = input_df['BEGIN_YEARMONTH'].astype('str').str[0:4]
detail_df['start_month'] = input_df['BEGIN_YEARMONTH'].astype('str').str[4:6]
detail_df['start_day'] = input_df['BEGIN_DAY']
detail_df['injuries'] = input_df['INJURIES_DIRECT'] + input_df['INJURIES_INDIRECT']
detail_df['deaths'] = input_df['DEATHS_DIRECT'] + input_df['DEATHS_INDIRECT']
detail_df['source'] = input_df['SOURCE']
detail_df['fujita_scale'] = input_df['TOR_F_SCALE']

print(detail_df['state_fips'].sort_values().unique())

# machine learning
# split year in two halves
# see if you can predict the dates of tornadoes

# PLAN
# make new columns - currenc

# regression
# see if you can predict the number of tornadoes and other extreme weather events happening in the future

import xarray as xr

ds = xr.open_dataset('ncdd-202512-grd-prelim.nc')
df = ds.to_dataframe()
print(df)