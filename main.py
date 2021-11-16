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
def main():
    print("Program has started...")
    pre_processing()
    print("Program has finished...")
    return

# Preprocessing function starts here
def pre_processing():
    print("Started preprocessing of data...")
    # Load Data from CSV file of dataset 1
    weather_dataframe = pd.read_csv('weather.csv')
    # data = weather_dataframe.values.tolist()
    print("Finished preprocessing data...")


if __name__ == "__main__":
    main()
