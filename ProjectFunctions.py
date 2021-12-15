# This file regroups all the functions we use in the project

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold,GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score


##########################################################################################################################################
############################################### Function to import the dataset ###########################################################
##########################################################################################################################################
### by CHILD-JAUVERT Victor

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
    
    
##########################################################################################################################################
############################################### Function to clean the dataset ############################################################
##########################################################################################################################################
### by CHILD-JAUVERT Victor

def cleanData(data):
    """
    This function takes in entry the dataset we created before with function importData.
    It drops useless columns, cleans dirtiness of the data, replaces missing values, centers & normalizes the data.
    The cleaning is done in situ.
    """
    
    # Remove "ground-truth" columns and store their values aside
    if data.columns[0] == 'variance' :
        y = data["class"]
        data = data.drop(columns="class")

    if data.columns[0] == 'id' :
        data.drop('id',inplace=True,axis=1)

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
    #for col in data_not_num_col:
        #print('{} has {} values'.format(col,data[col].unique()),'\n')

    if data.columns[0] == 'age' :

        #Corrects the input errors in the non numerical variables
        data['pcv'].replace(to_replace={'\t43':'43','\t?':np.nan,'?':np.nan},inplace=True)
        data['wc'].replace(to_replace={'\t6200':'6200','\t8400':'8400','\t?':np.nan,'?':np.nan},inplace=True)
        data['rc'].replace(to_replace={'\t?':np.nan,'?':np.nan},inplace=True)
        
        # Convert them to numerical variables
        data['pcv'] = pd.to_numeric (data['pcv'])
        data['wc'] = pd.to_numeric (data['wc'])
        data['rc'] = pd.to_numeric (data['rc'])
        
        # Add them to our list of numerical variables and remove them from the other list
        data_num_col.append('pcv')
        data_num_col.append('wc')
        data_num_col.append('rc')
        data_not_num_col.remove('pcv')
        data_not_num_col.remove('wc')
        data_not_num_col.remove('rc')
        
        # Corrects the input errors in the non numerical variables
        data['dm'].replace(to_replace={' yes':'yes','\tno':'no','\tyes':'yes'},inplace=True)
        data['cad'].replace(to_replace={'\tno':'no'},inplace=True)
        data['classification'].replace(to_replace={'ckd\t':'ckd'},inplace=True)
        
        # No more dirtiness in non numerical data
        #for col in kdn_not_num_col:
            #print('{} has {} values'.format(col,kidney[col].unique()),'\n')
        
        # Remove "ground-truth" columns and store their values aside
        y = data["classification"]
        data = data.drop(columns="classification")
        data_not_num_col.remove('classification')
        # One-hot encodes the ground truth
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Check the number of missing values in our data for each variable
    if data.isna().sum(axis=0).any():
        
        # Replace missing values by average values for numerical variables
        for col in data_num_col:     
            data[col].fillna(data[col].mean(),inplace=True)

        # Replace missing values by the most frequent value for non numerical variables.
        # To preserve the categorical aspect of the variables 
        for col in data_not_num_col:
            data[col].fillna(data[col].value_counts().index[0],inplace=True)

    # Center and normalize the data
    for col in data_num_col:
        data[col] = (data[col] - data[col].mean()) / (data[col].std())

    # One-hot encodes non numerical features
    le = LabelEncoder()
    for col in data_not_num_col:
        data[col]=le.fit_transform(data[col])
        
    # Return the data and the "ground-truth"
    return data, y
        
    
##########################################################################################################################################
############################################### Function to split the dataset ############################################################
##########################################################################################################################################
### by DUSSOLLE Antoine

def splitData(dataset,y,kfold=False,n_splits=5):
    
    """ 
     arguments : 
         dataset : a pandas Dataframe containing the data. 
         y : one-dimensional ndarray, ground truth (correct) labels.
         kfold : a boolean indicating whether the dataset needs to be split between training and test sets only (false), 
         or if the training set also needs to be split for cross-validation. 
         n_splits : number of fold (useful only if K-Fold activated).
         
     returns :
         X_train,X_test : pandas Dataframes corresponding to the train-test split of the input dataset.
         y_train,y_test : one-dimensional ndarray corresponding to the train-test split of the ground truth (correct) labels.
         if K-Fold : 
             K_Fold_X_train,K_Fold_X_valid : Lists containing the KFold splits of the training and validation datasets.
             K_Fold_y_train,K_Fold_y_valid :  Lists containing the KFold splits of the training and validation labels.
             X_test : pandas Dataframe corresponding to the test split of the input dataset.
             y_test : one-dimensional ndarray corresponding to the test split of the ground truth (correct) labels.    
    """
    
    
    #Split the dataset between training set and test set
    X_train,X_test,y_train,y_test = train_test_split(dataset,y,test_size = 0.3)
    
    #Training set preparation for 10-Fold cross-validation
    if kfold==True:

        # Shuffling there is not mandatory since train_test_split already shuffle the data 
        # but we still decided to use it
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=34) 

        K_Fold_X_train = []
        K_Fold_X_valid = []
        K_Fold_y_train = []
        K_Fold_y_valid = []

        for train_index, test_index in kf.split(X_train):
            X_cross_train, X_valid = X_train.iloc[train_index], X_train.iloc[test_index]
            y_cross_train, y_valid = y_train.iloc[train_index], y_train.iloc[test_index]

            K_Fold_X_train.append(X_cross_train)
            K_Fold_X_valid.append(X_valid)
            K_Fold_y_train.append(y_cross_train)
            K_Fold_y_valid.append(y_valid)

        return K_Fold_X_train,K_Fold_X_valid,K_Fold_y_train,K_Fold_y_valid,X_test,y_test
    else:
        return X_train,X_test,y_train,y_test     

    
##########################################################################################################################################
############################################### Functions to train the models ############################################################
##########################################################################################################################################       


### Train function for K-Nearest Neighbors model ###
### by CLAIR Robin

def trainKnn(X_train,y_train,crossval=False,n_neighbors=np.arange(1,25),cv=5):
    
    """
    Train a K-Nearest Neighbors model using cross-validation for the choice of K.
    
     arguments :
         X_train : pandas Dataframe, training data.
         y_train : one-dimensional ndarray, target values for training data.
         crossval : a boolean indicating whether or not cross-validation must be performed to find the best number of neighbors.
         n_neigbors : list of the value of neighbor's number to be tested.
         n_splits : number of fold for cross-validation.
         
     returns :
         knn_gscv.best_estimator_ : fit model with the most accurate number of neighbor value
         knn_gscv.best_params_ : top performing number of neigbor value
         knn_gscv.best_score_ : mean score for the top performing number of neigbor value
    """
    
    #Train the KNN classifier.
    
    #initialize a KNeighborsClassifier with n_neighbors default value (=5)
    knn = KNeighborsClassifier()
    
    if crossval == True :
        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': (n_neighbors)}

        #use gridsearch to test all values for n_neighbors using cross-validation with K-fold of value cv
        knn_gscv = GridSearchCV(knn, param_grid, cv=cv)

        #fit model to data
        knn_gscv.fit(X_train, y_train)

        return knn_gscv.best_estimator_,knn_gscv.best_params_,knn_gscv.best_score_ 
    else:
        knn.fit(X_train,y_train)
        return knn

    
### Train function for Random Forest model ###
### by DUSSOLLE Antoine

def trainRfc(X_train,y_train,featureselection=False,t=0.15,X_test=None):
    
    """ 
    Train a Random forest model using feature selection on a first random forest classifier
    to produce a final one only based on the most relevant features.
    
     arguments : 
         X_train : pandas Dataframe, training data.
         y_train : one-dimensional ndarray, target values for training data.
         featureselection : a boolean indicating whether or not feature selection needs to be performed.
         if feature selection : 
             threshold : float, threshold for the importance a feature needs to meet in order to be selected.
             X_test : pandas Dataframe, testing data (to be updated if only selected features are considered).
         
     returns :
         rfc : fit model
    """
    
    # Create a random forest classifier
    rfc = RandomForestClassifier(n_estimators=1000,random_state = 22)
    
    # Train the classifier
    rfc = rfc.fit(X_train,y_train)
    
    if featureselection==True:
        
        # Create a selector object that will use the random forest classifier to identify
        # Features that have an importance of more than 'threshold'
        rfc_sfm = SelectFromModel(rfc, threshold=t)
        #train the selector
        rfc_sfm.fit(X_train,y_train)
        
        # Plot the feature importance 
        plt.figure(figsize=(12,3))
        features = X_train.columns.values.tolist()
        importance = rfc.feature_importances_.tolist()
        feature_series = pd.Series(data=importance,index=features)
        feature_series.plot.bar()
        plt.title('Feature Importance')
        
        # Print the most important features that will be used for the improved model implementing feature selection
        selected_features = []
        for index, value in feature_series.items():
            if value>t:
                selected_features.append(index)
        print('The most relevant features are :',selected_features)
       
        # Transform the data to create a new dataset containing only the most relevant features
        #note: If done, it must also be done on the testing after used later on !
        X_selected_train = rfc_sfm.transform(X_train)
        X_selected_test = rfc_sfm.transform(X_test)

        # Create a new random forest classifier for the most important features
        rfc_selected = RandomForestClassifier(n_estimators=1000, random_state=0)

        # Train the new classifier on the new dataset containing the most important features
        rfc_selected.fit(X_selected_train, y_train)
        return rfc_selected,X_selected_test
    
    else:
        return rfc

##########################################################################################################################################
############################################### Functions to test the model ##############################################################
##########################################################################################################################################      
### by DE ROMEMONT Charlotte

def testModel(model, X_test, y_test):

    """ 
    This function test the model of classification we have used previously.
    It takes for arguments the test set and returns the classification report which includes precision, recall, f1 score, support
    It also returns the accuracy score of the model.
    """

    # Validation for the chosen model of classification.
    model_pred = model.predict(X_test)
    print("This is the classification report : ", classification_report(y_test, model_pred))
    print("This is the accuracy score : ", accuracy_score(y_test, model_pred))
