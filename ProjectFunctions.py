# Function to import the dataset

import pandas as pd 


def importData(file):
    #Reading dataset

    if file == 'data_banknote_authentication.txt' :
        # Reading the banknotes dataset
        banknotes = pd.read_csv(file, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

        # Indicates the numbers of rows and columns present in our dataset
        banknotesShape = banknotes.shape

        # Showcases the first rows of our dataset
        banknotesHead = banknotes.head()

        return banknotes, banknotesShape, banknotesHead
    
    elif file == 'kidney_disease.csv' :
        # Reading the chronic kidney disease dataset
        kidney = pd.read_csv(file)

        # Indicates the numbers of rows and columns present in our dataset
        kidneyShape = kidney.shape

        # Showcases the first rows of our dataset
        kidneyHead = kidney.head()

        return kidney, kidneyShape, kidneyHead

    else :
        return "dataset not recognized"
