# IMPORTING DEPENDENCIES
import numpy as np
import pandas as pd
import pickle

# standardScaler is used to standardize the data to a common range
from sklearn.preprocessing import StandardScaler

# train_test_split is used to split the data into training data and testing data
from sklearn.model_selection import train_test_split

# importing the model SVM
from sklearn import svm
from sklearn.metrics import accuracy_score

############################################################################
# DATA COLLECTION AND ANALYSIS

# Load the diabetes dataset to a pandas Dataframe.
# The dataset contains details of Females only.
# The outcome contains 0 or 1 where 1 indicates diabetic and 0 indicates non-diabetic.
diabetes_dataset = pd.read_csv("Flask\diabetes.csv")

# print the first 5 rows of the dataframe
# print(diabetes_dataset.head())

# print the number of rows and columns of the dataframe
# print(diabetes_dataset.shape)

# gives the statistical measures of the data
# print(diabetes_dataset.describe())

# number of diabetes data and non-diabetic data
# print(diabetes_dataset['Outcome'].value_counts())

# finding mean of each column(feature) based on diabetic(1) and non-diabetic(0) outcomes
# most important statistics which the ML model uses to predict the output.
# print(diabetes_dataset.groupby('Outcome').mean())

# separate the columns of the dataset such that the features and final outcome (0 or 1)
# of the features are separately stored
# while using drop function, axis value should be specified. axis=1 if we are dropping
# a column and axis=0 if we are dropping a row
# X stores all the columns of the dataset except Outcome and Y stores the Outcome values only.
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardizing the data
# the range of each feature will be different. Eg: glucose range is (0-200), age(33 to 81)
# standardization is done to convert these values in same range to build a efficient ML model
scaler = StandardScaler()                   # creating an instance
scaler.fit(X)
standardized_data = scaler.transform(X)     # data is transformed into a common range
# print(standardized_data)                    # all data are converted into the range of 0 and 1
X = standardized_data                       # storing it back into X variable

####################################################
# Split the data into training and testing data
# test_size 0.2 represents 20% data is given for test data nd the remaining for train data
# stratify by Y means, Y has the value as 0 or 1. stratifying makes sure that the same
# proportion of 0 and 1 are given to test and train sets. If not mentioned there is a possibility that
# the entire diabetes amy go to test and entire non-diabetes may go to training or vice-versa.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
#print(X.shape, X_train.shape, X_test.shape)

#####################################################
# training the model
# we are importing Support Vector Classifier from svm. kernel=linear implies we are using a linear SVM
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

######################################################
################### MODEL EVALUATION  ###################

# accuracy score on TRAINING DATA
# this will predict the labels(Y value) for all the X_train data and store it in the variable
# using the accuracy_score function, we are comparing the predicted value for Y with the original value
# accuracy score more than 75% is pretty good. but since we are using less data, it might be less
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print('Accuracy score on training data:', training_data_accuracy)

# accuracy score on TEST DATA
# the accuracy score on test data indicates whether the model is over trained.
# the model will be over-trained on the training data that it cannot perform well on the test data
# in that case, accuracy score on training data will be very high while it will be very less on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print('Accuracy score on test data:', test_data_accuracy)

################################################################
###################    MAKING A PREDICTIVE SYSTEM    ################

# INPUT data
# input_data1 = input('Enter the details of the patient: ')
# input_data = input_data1.split(",")
# input_data = (0,118,84,47,230,45.8,0.551,31)

# CONVERT TO NUMPY
# converting the input_data which is a list to numpy array since its more easy and efficient
# input_data_as_numpy_array = np.asarray(input_data)

# RESHAPE the array as we are predicting for one instance
# the dataset we are using contains 768 data points. So the model will expect 768 data-points
# but we are giving only one instance. Reshaping will tell the model that we need prediction for
# only one data point
# the length of the dimension set to -1 is automatically determined
# from the specified values of other dimensions
# input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# STANDARDIZE THE INPUT DATA
# this data cannot be fed to the model directly because we trained the model with standardized data
# so we need to standardize the input data before feeding it to the model
# we created an instance of StandardScalar, 'scaler', previously to which we fitted the training data(X)
# we need to use the same instance here but we need not fit it again. We need to transform it based
# on the scaler
# std_data = scaler.transform(input_data_reshaped)
# print(std_data)

# MAKING THE PREDICTION
# prediction = classifier.predict(std_data)
# print(prediction)                       # the model has to print 0 since the input is of a non-diabetic person

# the prediction variable is a list with 1 element
# if prediction[0] == 0:
#     print('Yaay! The person is non-diabetic')
# else:
#     print('Oops! The person is diabetic')


pickle.dump(classifier,open('model.pkl', 'wb'))
new_model = pickle.load(open('model.pkl', 'rb'))