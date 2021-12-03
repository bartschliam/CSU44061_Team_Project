import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import neighbors
from itertools import repeat

# Main function starts here ->
def main(): # TK, LB
    print('Program has started...')
    pre_processing() # pre process the data
    linear = True # run linear regression boolean
    lasso = True # run lasso regression boolean
    ridge = True # run ridge regression boolean
    knn = True # run knn regression boolean
    dummy = True # run dummy regressions boolean
    compare = True # run plot and compare boolean
    methods(linear, lasso, ridge, knn, dummy, compare) # linear, lasso, ridge and knn regression, dummy and compare
    print('...Program has finished')
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
    daylight_saving_time_index = 2137
    wea_date = wea.iloc[relevant_index:,0] # load 1st column, date
    wea_date = wea_date.drop(wea_date.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_rain = wea.iloc[relevant_index:,2] # load 3rd column, precipitation amount (rain)
    wea_rain = wea_rain.drop(wea_rain.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_temperature = wea.iloc[relevant_index:,4] # load 5th column, temperature
    wea_temperature = wea_temperature.drop(wea_temperature.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_humidity = wea.iloc[relevant_index:,9] # load 10th column, relative humidity
    wea_humidity = wea_humidity.drop(wea_humidity.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_wind_speed = wea.iloc[relevant_index:,12] # load 13th column, mean wind speed
    wea_wind_speed = wea_wind_speed.drop(wea_wind_speed.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_wind_direction = wea.iloc[relevant_index:,14] # load 15th column, wind direction
    wea_wind_direction = wea_wind_direction.drop(wea_wind_direction.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_sun_duration = wea.iloc[relevant_index:,17] # load 18th column, sun duration
    wea_sun_duration = wea_sun_duration.drop(wea_sun_duration.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_visibility = wea.iloc[relevant_index:,18] # load 19th column, visibility
    wea_visibility = wea_visibility.drop(wea_visibility.index[daylight_saving_time_index]) # drop daylight saving time error
    wea_cloud_amount = wea.iloc[relevant_index:,20] # load 21st column, cloud amount
    wea_cloud_amount = wea_cloud_amount.drop(wea_cloud_amount.index[daylight_saving_time_index]) # drop daylight saving time error

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
    
    frame = { 'Date & Time': wea_date, 'Rain': wea_rain, 'Temperature': wea_temperature, 'Humidity': wea_humidity, 
        'Wind Speed': wea_wind_speed, 'Wind Direction': wea_wind_direction, 'Sun Duration': wea_sun_duration, 
        'Visibility': wea_visibility, 'Cloud Amount': wea_cloud_amount, 'Total Count': total_count.tolist() } # combine columns into a frame
    result = pd.DataFrame(frame) # add frame to dataframe
    result['Date & Time'] = result['Date & Time'].str.replace('-','/') # replace the dashes with forward slashes so the two formats are the same
    result['Date & Time'] = result['Date & Time'].str.replace(r'(\d{2}):(\d{2}):(\d{2})', r'\1:\2', regex=True) # delete the :ss since they are all 00 seconds which doesn't provide more information
    result.to_csv('results.csv', index=False) # write dataframe to csv file 
    print('Finished preprocessing data...')

def methods(li, la, ri, kn, du, co): # CT, TK
    print('STARTING SELECTED METHODS')
    
    
    # ----------------------------------------------------------------------------#
    ############################ Initialization and SET-UP ###################################
    # ----------------------------------------------------------------------------#
    result = pd.read_csv('results.csv') # read preprocessed data
    y = result.iloc[:,9] # read the count (output)
    X = result.iloc[:,0:9] # read in all the other columns (inputs)
    X['Date & Time'] = X['Date & Time'].str[11:-3] # truncate date and time to only the hours

    
    c_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] # A list of different CValues
    k_values = list(range(1, 21)) # create a list from 1 to 20 for different k values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # split data into training and test data
    la_mean = [] # mean error lasso
    la_std = [] # std error lasso
    ri_mean = [] # mean error ridge
    ri_std = [] # std error ridge
    knn_mean = [] # mean error knn
    knn_std = [] # std error knn
    
    models = ['Linear\nRegression', 'Lasso\nRegression', 'Ridge\nRegression', 'KNN\nRegression', 'Mean\nDummy', 'Median\nDummy'] # set x labels for bar plot
    final_compare_x = np.arange(len(models)) # evenly spaced intervals
    final_compare_mean = [] # init mean values for final compare
    final_compare_std = [] # init std values for final compare
    kf = KFold(n_splits=5) # 5 splits for kfold
    li_model = LinearRegression() # initialize linear regression model
    dummy_mean = DummyRegressor(strategy='mean') # dummy mean model
    dummy_median = DummyRegressor(strategy='median') # dummy median model

    # ----------------------------------------------------------------------------#
    ############################ Linear Regression ################################
    # ----------------------------------------------------------------------------#
    if(li is True): # if boolean for linear regression is set
        print('Starting linear regression...')
        li_error = [] # linear regression error
        for train, test in kf.split(x_train): # perform 5-fold cross validation on training data
            li_model.fit(x_train.iloc[train], y_train.iloc[train]) # fit the current's fold training data
            y_pred_li = li_model.predict(x_train.iloc[test]) # predict using current's fold test data
            li_error.append(metrics.mean_squared_error(y_train.iloc[test], y_pred_li)) # add error to temp array
        li_model.fit(x_train, y_train) # fit the model on training data
        final_y_pred_li = li_model.predict(x_test) # predict on test data
        li_rmse_error = metrics.mean_squared_error(y_test, final_y_pred_li) # get RMSE from the actual outputs
        li_error.append(li_rmse_error) # add last prediction on test data to array of errors
        final_compare_mean.append(np.mean(li_error)) # add mean to final mean
        final_compare_std.append(np.std(li_error)) # add std to final std

        print("Coefficients for the input features: ")
        print(pd.DataFrame(li_model.coef_, X.columns, columns=['Coeff'])) # save coefficients of each input feature
        print('...Finished linear regression')
    
    # ----------------------------------------------------------------------------#
    ############################ Lasso Regression ################################
    # ----------------------------------------------------------------------------#
    if(la is True): # if boolean for lasso regression is set
        print('Starting lasso regression...')
        for c in c_values: # cross validation to select hyper parameter c
            la_error = [] # lasso error

            currentAlpha = 1/(2*c) # alpha value for current model using current c value
            la_model = Lasso(alpha=currentAlpha) # lasso model
            for train, test in kf.split(x_train): # use 5 fold cross validation on training data
                la_model.fit(x_train.iloc[train], y_train.iloc[train]) # fit the model on training fold
                y_pred_la = la_model.predict(x_train.iloc[test]) # predict on test fold
                la_error.append(metrics.mean_squared_error(y_train.iloc[test], y_pred_la)) # compare prediction and report error
            la_mean.append(np.mean(la_error)) # add mean of errors
            la_std.append(np.std(la_error)) # add std of errors
            

            # Choosing the best value of C for comparing the Lasso Model with other models
            best_C_val_lasso = 10   #Selected the best value of C
            if(c == best_C_val_lasso): # select c value with best results
                final_compare_mean.append(np.mean(la_error)) # add to final mean comparison
                final_compare_std.append(np.std(la_error)) # add to final  std comparison
        print('...Finished lasso regression')

    # ----------------------------------------------------------------------------#
    ############################ Ridge Regression ################################
    # ----------------------------------------------------------------------------#
    if(ri is True): # if ridge boolean is set
        print('Starting ridge regression...')
        for c in c_values: # cross validation to select hyper parameter c
            ri_error = [] # ridge error
            
            currentAlpha = 1/(2*c) # alpha value for current model using current c value
            ri_model = Ridge(alpha=currentAlpha) # ridge model
            for train, test in kf.split(x_train): # use 5 fold cross validation on training data
                ri_model.fit(x_train.iloc[train], y_train.iloc[train]) # fit the model on training fold
                y_pred_ri = ri_model.predict(x_train.iloc[test]) # predict on test fold
                ri_error.append(metrics.mean_squared_error(y_train.iloc[test], y_pred_ri)) # compare prediction and report error
            ri_mean.append(np.mean(ri_error)) # add mean of errors
            ri_std.append(np.std(ri_error)) # add std of errors
            

            # Choosing the best value of C for comparing the Ridge Model with other models
            best_C_val_ridge = 0.1  #Selected the best value of C for Ridge Regression
            if(c == best_C_val_ridge): # select c value with best results
                final_compare_mean.append(np.mean(ri_error)) # add to final mean comparison
                final_compare_std.append(np.std(ri_error)) # add to final std comparison
        print('...Finished ridge regression')


    # ----------------------------------------------------------------------------#
    ############################ KNN Regression ###################################
    # ----------------------------------------------------------------------------#    
    if(kn is True): # if knn boolean is set
        print('Starting knn regression...')
        for k in k_values: # for each k value
            kn_error = [] # kn_error array to store errors
            knn_model = neighbors.KNeighborsRegressor(n_neighbors=k) # init model
            for train, test in kf.split(x_train): # perform 5-fold cross validation on training data
                knn_model.fit(x_train.iloc[train], y_train.iloc[train]) # fit model on training data
                y_pred_knn = knn_model.predict(x_train.iloc[test]) # predict on test data
                kn_error.append(metrics.mean_squared_error(y_train.iloc[test], y_pred_knn)) # add error to kn_error array
            knn_mean.append(np.mean(kn_error)) # add mean of estimates
            knn_std.append(np.std(kn_error)) # add std of estimates
            if(k == 10): # select best k value
                final_compare_mean.append(np.mean(kn_error)) # add error to final mean comparison
                final_compare_std.append(np.std(kn_error)) # add error to final std comparison
        print('...Finished knn regression')

    # ----------------------------------------------------------------------------#
    ############################ Dummy Regressors #################################
    # ----------------------------------------------------------------------------#
    if(du is True): # if dummy boolean is set
        dummy_mean.fit(x_train, y_train) # fit mean model
        y_pred_mean = dummy_mean.predict(x_test) # make prediction
        final_compare_mean.append(np.mean(metrics.mean_squared_error(y_pred_mean, y_test))) # mean rmse dummy mean
        final_compare_std.append(np.std(metrics.mean_squared_error(y_pred_mean, y_test))) # std rmse dummy mean

        dummy_median.fit(x_train, y_train) # fit median model
        y_pred_median = dummy_median.predict(x_test) # make prediction
        final_compare_mean.append(np.mean(metrics.mean_squared_error(y_pred_median, y_test))) # mean rmse dummy median
        final_compare_std.append(np.std(metrics.mean_squared_error(y_pred_median, y_test))) # std rmse dummy median
    # ----------------------------------------------------------------------------#
    ############################ Plot and Compare #################################
    # ----------------------------------------------------------------------------#
    if(co is True): # if compare boolean is set
        print('Starting plotting and comparing...')

        plt.scatter(X['Date & Time'], y, s=1) # plot time against cycle count
        plt.xlabel('Hour of the day') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Hour of the day') # set title
        plt.show() # show plot

        plt.scatter(X['Rain'], y, s=1) # plot rain against cycle count
        plt.xlabel('Rain amount') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Rain amount') # set title
        plt.show() # show plot

        plt.scatter(X['Temperature'], y, s=1) # plot temperature against cycle count
        plt.xlabel('Temperature') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Temperature') # set title
        plt.show() # show plot

        plt.scatter(X['Humidity'], y, s=1) # plot humidity against cycle count
        plt.xlabel('Humidity') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Humidity') # set title
        plt.show() # show plot

        plt.scatter(X['Wind Speed'], y, s=1) # plot wind speed against cycle count
        plt.xlabel('Wind speed') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Wind speed') # set title
        plt.show() # show plot

        plt.scatter(X['Wind Direction'], y, s=1) # plot wind direction against cycle count
        plt.xlabel('Wind direction') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Wind direction') # set title
        plt.show() # show plot

        plt.scatter(X['Sun Duration'], y, s=1) # plot sun duration against cycle count
        plt.xlabel('Sun Duration') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Sun Duration') # set title
        plt.show() # show plot

        plt.scatter(X['Visibility'], y, s=1) # plot visibility against cycle count
        plt.xlabel('Visibility') # set x axis
        plt.ylabel('Cycle Count') # set y axis
        plt.title('Cycle count vs Visibility') # set title
        plt.show() # show plot

        plt.scatter(X['Cloud Amount'], y, s=1) # plot cloud amount against cycle count
        plt.xlabel('Cloud Amount') # set x axis
        plt.ylabel('Cycle count') # set y axis
        plt.title('Cycle count vs Cloud Amount') # set title
        plt.show() # show plot

        plt.errorbar(np.log10(c_values), la_mean, yerr= la_std, label='Lasso Model') # plot error bar for lasso regression
        plt.errorbar(np.log10(c_values), ri_mean, yerr= ri_std, label='Ridge Model') # plot error bar for ridge regression
        plt.xlim(-7.5, 7.5) # limit x values for better viewing of plot
        plt.xlabel('Log10(C) values') # set x axis
        plt.ylabel('Root mean squared error') # set y axis
        plt.title('Log10(C) vs Root mean squared error \nfor Lasso and Ridge regression') # title
        plt.legend() # show legend
        plt.show() # show plot

        plt.errorbar(k_values, knn_mean, yerr=knn_std, label='KNN regression') # plot errorbar for knn
        plt.legend() # plot legend
        plt.xlabel('k values') # set x label
        plt.ylabel('Root mean square error') # set y label
        plt.title('Root Mean Squared Error (RMSE) \nfor KNN Regression vs different k values') # set title
        plt.show() # show plot

        fig, ax = plt.subplots() # create a subplot
        ax.bar(final_compare_x, final_compare_mean, yerr=final_compare_std, align='center', alpha=0.5, ecolor='black', capsize=10) # create bar plot
        ax.set_ylabel('Root squared mean error') # set y axis
        ax.set_xticks(final_compare_x) # set x axis
        ax.set_xticklabels(models) # set x axis labels
        ax.set_title('Root squared mean error comparison of different regression models') # set title
        ax.yaxis.grid(True) # show grid
        plt.show() # show plot
        print('Finished plotting and comparing...')

    print('...Finished methods')

if __name__ == '__main__':
    main()