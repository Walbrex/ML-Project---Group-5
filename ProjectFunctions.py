# This file regroups all the functions we use in the project

import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Function to import the dataset 

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
    
# Function to clean the dataset

def cleanData(data):
    """
    This function takes in entry the dataset we created before with function importData.
    It drops useless columns, cleans dirtiness of the data, replaces missing values, centers & normalizes the data.
    The cleaning is done in situ.
    """
    
    # Remove "ground-truth" columns and store their values aside

    if data.columns[0] == 'variance' :
        banknotes = data
        y_bkn = banknotes["class"]
        banknotes = banknotes.drop(columns="class")

    if data.columns[0] == 'id' :
        kidney = data
        kidney.drop('id',inplace=True,axis=1)

    # Check if all variables are accounted as numerical ones 

    data_num_col = []
    data_not_num_col = []
    
    for col in data.columns:
        if data[col].dtype==np.int64 or data[col].dtype==np.float64:
            data_num_col.append(col)
        else:
            data_not_num_col.append(col)
    
    #print('This is the numerical columns of our dataset :','\n',data_num_col)
    #print('This is the non numerical columns of our dataset :','\n',data_not_num_col)

    # Highlighting dirtiness in non numerical data
    
    for col in data_not_num_col:
        print('{} has {} values'.format(col,data[col].unique()),'\n')

    if data.columns[0] == 'id' :

        #Corrects the input errors in the non numerical variables
        kidney['pcv'].replace(to_replace={'\t43':'43','\t?':np.nan,'?':np.nan},inplace=True)
        kidney['wc'].replace(to_replace={'\t6200':'6200','\t8400':'8400','\t?':np.nan,'?':np.nan},inplace=True)
        kidney['rc'].replace(to_replace={'\t?':np.nan,'?':np.nan},inplace=True)
        
        # Convert them to numerical variables
        kidney['pcv'] = pd.to_numeric (kidney['pcv'])
        kidney['wc'] = pd.to_numeric (kidney['wc'])
        kidney['rc'] = pd.to_numeric (kidney['rc'])
        
        # Add them to our list of numerical variables and remove them from the other list
        data_num_col.append('pcv')
        data_num_col.append('wc')
        data_num_col.append('rc')
        data_not_num_col.remove('pcv')
        data_not_num_col.remove('wc')
        data_not_num_col.remove('rc')
        
        # Corrects the input errors in the non numerical variables
        kidney['dm'].replace(to_replace={' yes':'yes','\tno':'no','\tyes':'yes'},inplace=True)
        kidney['cad'].replace(to_replace={'\tno':'no'},inplace=True)
        kidney['classification'].replace(to_replace={'ckd\t':'ckd'},inplace=True)

        # No more dirtiness in non numerical data
        #for col in kdn_not_num_col:
            #print('{} has {} values'.format(col,kidney[col].unique()),'\n')
        
        # Remove "ground-truth" columns and store their values aside
        y_kdn = kidney["classification"]
        kidney = kidney.drop(columns="classification")


    # Check the number of missing values in our data for each variable
    
    if data.isna().sum(axis=0).any():
        
        # Replace missing values by average values for numerical variables
        for col in data_num_col:     
            kidney[col].fillna(kidney[col].mean(),inplace=True)

        # Replace missing values by the most frequent value for non numerical variables.
        # To preserve the categorical aspect of the variables 
        for col in data_not_num_col:
            kidney[col].fillna(kidney[col].value_counts().index[0],inplace=True)

    # Center and normalize the data

    for col in data_num_col:
        data[col] = (data[col] - data[col].mean()) / (data[col].std())


    # One-hot encodes non numerical features
    le = LabelEncoder()
    for col in data_not_num_col:
        data[col]=le.fit_transform(data[col])
