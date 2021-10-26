print('Program Starts')

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





# MAIN FUCTION STARTS HERE ->

# Load Data from CSV file of dataset 1
weather_dataframe = pd.read_csv('weather.csv')
# data = weather_dataframe.values.tolist()


# FOR PART 1
print("Program Ends")
