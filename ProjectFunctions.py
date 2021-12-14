# This file regroups all the functions we use in the project

# Function to import the dataset

import pandas as pd 


def importData(file):
    """
    This function takes the name of the file in entry, then read the file and import a dataframe.
    It returns the dataframe, its shape and the first rows/columns.
    """

    if file == 'data_banknote_authentication.txt' or file[len(file)-3:] == 'txt' :
        # Reading the banknotes dataset
        banknotes = pd.read_csv(file, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

        # Indicates the numbers of rows and columns present in our dataset
        banknotesShape = banknotes.shape

        # Showcases the first rows of our dataset
        banknotesHead = banknotes.head()

        return banknotes, banknotesShape, banknotesHead
    
    elif file == 'kidney_disease.csv' or file[len(file)-3:] == 'csv' :
        # Reading the chronic kidney disease dataset
        kidney = pd.read_csv(file)

        # Indicates the numbers of rows and columns present in our dataset
        kidneyShape = kidney.shape

        # Showcases the first rows of our dataset
        kidneyHead = kidney.head()

        return kidney, kidneyShape, kidneyHead

    else :
        return "dataset not recognized"
