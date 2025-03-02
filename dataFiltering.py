import pandas as pd

# Load in cleaned datasets from csv files
la_max_temp_cleaned = pd.read_csv("cleaned data/LA_max_temp_cleaned.csv")
la_min_temp_cleaned = pd.read_csv("cleaned data/LA_min_temp_cleaned.csv")
la_precipitation_cleaned = pd.read_csv("cleaned data/la_precipitation_cleaned.csv")
la_drought_cleaned = pd.read_csv("cleaned data/LA_drought_cleaned.csv")
max_wind_speed_cleaned = pd.read_csv("cleaned data/cal adapt wind speed chart monthly.csv")
historical_perimeter_cleaned = pd.read_csv("cleaned data/Historical_Perimeter_cleaned.csv")
# damage_inspection_cleaned = pd.read_csv("cleaned data/FIRE_Damage_Inspection_cleaned.csv")
damage_inspection_cleaned_post = pd.read_csv("cleaned data/POSTFIRE_MASTER_DATA_columns_removed.csv")   # use this one
fuel_moisture = pd.read_csv("raw data/fmr_1.csv")
# WORKING ON HISTORIC PERIMETERS BY YEAR

# Note: for all dates, either format as "year" (integer) or "month" (datetime object) [day = 1] or "date" (datetime object)

# LA_MAX_TEMP
# la_max_temp_cleaned already in correct "types" and no missing values w/in table
la_max_temp_cleaned = la_max_temp_cleaned[la_max_temp_cleaned.year >= 2000]     # Years 2000-2025
la_max_temp_cleaned.rename(columns={'Observed': 'Observed Max Temp'}, inplace=True)

# LA_MIN_TEMP
# la_min_temp_cleaned already in correct "types" and no missing values w/in table
la_min_temp_cleaned = la_min_temp_cleaned[la_min_temp_cleaned.year >= 2000]     # Years 2000-2025
la_min_temp_cleaned.rename(columns={'Observed': 'Observed Min Temp'}, inplace=True)

# LA_PRECIPITATION
la_precipitation_cleaned.rename(columns={'Observed': 'Max Observed Precipitation'}, inplace=True)

# LA_DROUGHT
la_drought_cleaned.Date = pd.to_datetime(la_drought_cleaned.Date)   # convert to datetime object
la_drought_cleaned.rename(columns={'Date': 'month'}, inplace=True)
move_date = la_drought_cleaned.pop('month')
la_drought_cleaned.insert(0, 'month', move_date)
la_drought_cleaned['year'] = la_drought_cleaned['month'].dt.year     # year column for merging

# MAX_WIND_SPEED
max_wind_speed_cleaned.year_month = pd.to_datetime(max_wind_speed_cleaned.year_month)
max_wind_speed_cleaned.rename(columns={'year_month': 'month', 'value': 'Max Wind Speed'}, inplace=True)
max_wind_speed_cleaned['year'] = max_wind_speed_cleaned['month'].dt.year
max_wind_speed_cleaned = max_wind_speed_cleaned[max_wind_speed_cleaned['year'] >= 2000]     # Years 2000-2025

# HISTORICAL PERIMETER
historical_perimeter_cleaned.drop(['STATE', 'DECADES'], axis=1, inplace=True)  # drop unnecessary columns
historical_perimeter_cleaned.ALARM_DATE = pd.to_datetime(historical_perimeter_cleaned.ALARM_DATE).dt.date
historical_perimeter_cleaned.ALARM_DATE = pd.to_datetime(historical_perimeter_cleaned.ALARM_DATE)   # datetime object
historical_perimeter_cleaned.CONT_DATE = pd.to_datetime(historical_perimeter_cleaned.CONT_DATE).dt.date
historical_perimeter_cleaned.CONT_DATE = pd.to_datetime(historical_perimeter_cleaned.CONT_DATE)   # datetime object
historical_perimeter_cleaned = historical_perimeter_cleaned.rename(columns={'YEAR_': 'year'})
historical_perimeter_cleaned['month'] = historical_perimeter_cleaned.ALARM_DATE.dt.to_period('M').dt.to_timestamp()

# DAMAGE_INSPECTION_CLEANED
damage_inspection_cleaned_post.drop(['State', 'County', '* City'], axis=1, inplace=True)
damage_inspection_cleaned_post['Incident Start Date'] = pd.to_datetime(damage_inspection_cleaned_post['Incident Start Date'])
damage_inspection_cleaned_post['* Damage'] = damage_inspection_cleaned_post['* Damage'].astype('category')
damage_inspection_cleaned_post['* Structure Type'] = damage_inspection_cleaned_post['* Structure Type'].astype('category')
damage_inspection_cleaned_post['Structure Category'] = damage_inspection_cleaned_post['Structure Category'].astype('category')
damage_inspection_cleaned_post.rename(columns={'Incident Start Date': 'Damage Inspection Date', '* Damage': 'Damage', '* Structure Type': 'Structure Type', '* CAL FIRE Unit': 'CAL FIRE Unit'}, inplace=True)
damage_inspection_cleaned_post['year'] = damage_inspection_cleaned_post['Damage Inspection Date'].dt.year
damage_inspection_cleaned_post['month'] = damage_inspection_cleaned_post['Damage Inspection Date'].dt.to_period('M').dt.to_timestamp()

# FUEL_MOISTURE
fuel_moisture.drop(['STID', 'FUEL_VARIATION', 'NAME', 'GACC'], axis=1, inplace=True)
fuel_moisture['DATE'] = pd.to_datetime(fuel_moisture['DATE'])
fuel_moisture.rename(columns={'DATE': 'Fuel Moisture Date'}, inplace=True)
fuel_moisture['year'] = fuel_moisture['Fuel Moisture Date'].dt.year
fuel_moisture['month'] = fuel_moisture['Fuel Moisture Date'].dt.to_period('M').dt.to_timestamp()


# Merging Datasets
# merge datasets that only consider year
df1 = la_max_temp_cleaned.merge(la_min_temp_cleaned, how='outer', left_on='year', right_on='year')    # max & min temps
df2 = df1.merge(la_precipitation_cleaned, how='outer', left_on='year', right_on='year')     # + precipitation
# merge datasets that also consider month
df3 = la_drought_cleaned.merge(max_wind_speed_cleaned, how='outer', left_on=['year', 'month'], right_on=['year', 'month'])   # drought index & wind speed
df4 = df3.merge(historical_perimeter_cleaned, how='outer', left_on=['year', 'month'], right_on=['year', 'month'])     # + historic fire data
df5 = df4.merge(damage_inspection_cleaned_post, how='outer', left_on=['year', 'month'], right_on=['year', 'month'])   # + damage inspection
df6 = df5.merge(fuel_moisture, how='outer', left_on=['year', 'month'], right_on=['year', 'month'])
# merge those groups
df7 = df2.merge(df6, how='outer', left_on='year', right_on='year')

merged_data = df7
merged_data.to_csv('filtered_data.csv')
