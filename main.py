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

# Main function starts here ->
def main(): # TK, LB
    print("Program has started...")
    pre_processing()
    print("Program has finished...")
    return

# Preprocessing function starts here
def pre_processing(): # LB
    print("Started preprocessing of data...")
    # Load data from datasets
    weather_dataframe = pd.read_csv('weather.csv')
    cycle_count_dataframe = pd.read_csv('cycle_counter.csv')
    # data = weather_dataframe.values.tolist()

    # load in the weather data and discard the first 23 rows since the data starts at index 23
    weather_date = weather_dataframe.iloc[23:,0] # load 1st column, date
    weather_rain = weather_dataframe.iloc[23:,2] # load 3rd column, precipitation amount (rain)
    weather_temperature = weather_dataframe.iloc[23:,4] # load 5th column, temperature
    weather_humidity = weather_dataframe.iloc[23:,9] # load 10th column, relative humidity
    weather_wind_speed = weather_dataframe.iloc[23:,12] # load 13th column, mean wind speed
    weather_wind_direction = weather_dataframe.iloc[23:,14] # load 15th column, wind direction
    weather_sun_duration = weather_dataframe.iloc[23:,17] # load 18th column, sun duration
    weather_visibility = weather_dataframe.iloc[23:,18] # load 19th column, visibility
    weather_cloud_amount = weather_dataframe.iloc[23:,20] # load 21st column, cloud amount
    print("Finished loading weather data...")


    #print(weather_dataframe.iloc[23,1])
    print("Finished preprocessing data...")


if __name__ == "__main__":
    main()
