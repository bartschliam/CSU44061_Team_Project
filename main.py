import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
warnings.filterwarnings('ignore')
import re

# Main function starts here ->
def main(): # TK, LB
    print('Program has started...')
    pre_processing() # pre process the data
    linear_regression() # 
    print('Program has finished...')
    return

# Preprocessing function starts here
def pre_processing(): # LB
    # Load data from datasets
    wea = pd.read_csv('weather.csv')
    count_2019 = pd.read_csv('2019_cycle_counter.csv')
    count_2020 = pd.read_csv('2020_cycle_counter.csv')
    count_2021 = pd.read_csv('2021_cycle_counter.csv')

    # load in the weather data and discard all the rows corresponding to dates before january 1st 2019
    relevant_index = 23 + 245448
    wea_date = wea.iloc[relevant_index:,0] # load 1st column, date
    wea_rain = wea.iloc[relevant_index:,2] # load 3rd column, precipitation amount (rain)
    wea_temperature = wea.iloc[relevant_index:,4] # load 5th column, temperature
    wea_humidity = wea.iloc[relevant_index:,9] # load 10th column, relative humidity
    wea_wind_speed = wea.iloc[relevant_index:,12] # load 13th column, mean wind speed
    wea_wind_direction = wea.iloc[relevant_index:,14] # load 15th column, wind direction
    wea_sun_duration = wea.iloc[relevant_index:,17] # load 18th column, sun duration
    wea_visibility = wea.iloc[relevant_index:,18] # load 19th column, visibility
    wea_cloud_amount = wea.iloc[relevant_index:,20] # load 21st column, cloud amount
    frame = { 'Date & Time': wea_date, 'Rain': wea_rain, 'Temperature': wea_temperature, 'Humidity': wea_humidity, 
        'Wind Speed': wea_wind_speed, 'Wind Direction': wea_wind_direction, 'Sun Duration': wea_sun_duration, 
        'Visibility': wea_visibility, 'Cloud Amount': wea_cloud_amount } # combine columns into a frame
    result = pd.DataFrame(frame) # add frame to dataframe
    result.to_csv('weather_result.csv') # add to file

    # load in cycle count data for 2019
    count_date_2019 = count_2019.iloc[:,0] # load 1st column, date
    count_grt_2019 = count_2019.iloc[:,1] # load 2nd column, Grove Road Totem (grt) location
    count_nsrs_2019 = count_2019.iloc[:,2] # load 3rd column, North Strand Rd S/B (nsrs) location
    count_nsrn_2019 = count_2019.iloc[:,3] # load 4th column, North Strand Rd N/B (nsrn) location
    count_cm_2019 = count_2019.iloc[:,4] # load 5th column, Charleville Mall (cm) location
    count_gs_2019 = count_2019.iloc[:,5] # load 6th column, Guild Street (gs) location
    total_count_2019 = count_2019.sum(axis=1) # calculate the total amount for each row

    # load in cycle count data for 2020
    count_date_2020 = count_2020.iloc[:,0] # load 1st column, date
    count_cm_2020 = count_2020.iloc[:,1] # load 2nd column, Charleville Mall (cm) location
    count_grt_2020 = count_2020.iloc[:,4] # load 5th column, Grove Road Totem (grt) location
    count_gs_2020 = count_2020.iloc[:,7] # load 8th column, Guild Street (gs) location
    count_nsrn_2020 = count_2020.iloc[:,10] # load 11th column, North Strand Rd N/B (nsrn) location
    count_nsrs_2020 = count_2020.iloc[:,11] # load 12th column, North Strand Rd S/B (nsrs) location
    column_list_2020 = list(count_2020) # get all the columns from the 2020 dataset
    column_list_2020.remove('Charleville Mall Cyclist IN') # remove IN as already counted
    column_list_2020.remove('Charleville Mall Cyclist OUT') # remove OUT as already counted
    column_list_2020.remove('Grove Road Totem OUT') # remove OUT as already counted
    column_list_2020.remove('Grove Road Totem IN') # remove IN as already counted
    column_list_2020.remove('Guild Street bikes IN-Towards Quays') # remove IN as already counted
    column_list_2020.remove('Guild Street bikes OUT-Towards Drumcondra') # remove OUT as already counted
    total_count_2020 = count_2020[column_list_2020].sum(axis=1) # sum up total of totals

    # load in cycle count data for 2021
    count_date_2021 = count_2021.iloc[:,0] # load 1st column, date
    count_cm_2021 = count_2021.iloc[:,1] # load 2nd column, Charleville Mall (cm) location
    count_d1_2021 = count_2021.iloc[:,4] # load 5th column, Drumcondra 1 (d1) location
    count_d2_2021 = count_2021.iloc[:,7] # load 8th column, Drumcondra 2 (d2) location
    count_grt_2021 = count_2021.iloc[:,10] # load 11th column, Grove Road Totem (grt) location
    count_nsrn_2021 = count_2021.iloc[:,13] # load 14th column, North Strand Rd N/B (nsrn) location
    count_nsrs_2021 = count_2021.iloc[:,14] # load 15th column, North Strand Rd S/B (nsrs) location
    count_r1_2021 = count_2021.iloc[:,15] # load 16th column, Richmond Street 1 (r1) location
    count_r2_2021 = count_2021.iloc[:,18] # load 19th column, Richmond Street 2 (r2) location
    column_list_2021 = list(count_2021) # all the columns from the 2021 dataset
    column_list_2021.remove('Charleville Mall Cyclist IN') # remove IN as already counted
    column_list_2021.remove('Charleville Mall Cyclist OUT') # remove OUT as already counted
    column_list_2021.remove('Drumcondra Cyclists 1 Cyclist IN') # remove IN as already counted
    column_list_2021.remove('Drumcondra Cyclists 1 Cyclist OUT') # remove OUT as already counted
    column_list_2021.remove('Drumcondra Cyclists 2 Cyclist IN') # remove IN as already counted
    column_list_2021.remove('Drumcondra Cyclists 2 Cyclist OUT') # remove OUT as already counted
    column_list_2021.remove('Grove Road Totem OUT') # remove OUT as already counted
    column_list_2021.remove('Grove Road Totem IN') # remove IN as already counted
    column_list_2021.remove('Richmond Street Cyclists 1 Cyclist IN') # remove IN as already counted
    column_list_2021.remove('Richmond Street Cyclists 1 Cyclist OUT') # remove OUT as already counted
    column_list_2021.remove('Richmond Street Cyclists 2  Cyclist IN') # remove IN as already counted
    column_list_2021.remove('Richmond Street Cyclists 2  Cyclist OUT') # remove OUT as already counted
    total_count_2021 = count_2021[column_list_2021].sum(axis=1) # sum up total of totals
    
    total_count_2021 = total_count_2021[:-503] # remove the last 503 rows since we only have weather data until 1/10/2021 00:00
    count_date_2021 = count_date_2021[:-503] # remove the last 503 rows since we only have weather data until 1/10/2021 00:00

    count_date_total = count_date_2019.append(count_date_2020.append(count_date_2021)) # combine 2019, 2020 and 2021 dates into one list
    total_count = total_count_2019.append(total_count_2020.append(total_count_2021)) # combine total count from 2019, 2020 and 2021
    frame = { 'Date & Time': count_date_total, 'Total Count': total_count } # create a frame with the date and total count
    result = pd.DataFrame(frame) # create a dataframe with our desired frame
    result['Date & Time'] = result['Date & Time'].str.replace('-','/') # replace the dashes with forward slashes so the two formats are the same
    result['Date & Time'] = result['Date & Time'].str.replace(r'(\d{2}):(\d{2}):(\d{2})', r'\1:\2', regex=True) # delete the :ss since they are all 00 seconds which doesn't provide more information
    result.to_csv('count_result.csv') # write dataframe to csv file    
    print('Finished preprocessing data...')

def linear_regression(): # LB
    print('Starting linear regression...')

    # Use cross validation to select hyperparameters
    # Compare performance against baseline predictors
    print('Finished linear regression...')

def lasso_regression():
    print('Starting lasso regression...')

    print('Finished lasso regression...')

def ridge_regression():
    print('Starting ridge regression...')

    print('Finished ridge regression...')

def knn():
    print('Starting knn...')

    print('Finished knn...')

if __name__ == '__main__':
    main()
