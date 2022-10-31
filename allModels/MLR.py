#Evaluation of the WEAPON feature from BPD Crime bpd_crime_dataframeset (pre-processed) using Multinomial Logistic Regression

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



# Load preprocessed bpd_crime_dataframeset
data = pd.read_csv('CrimaPreProcessed.csv')
print("\n\nFirst five rows of our Dataset: \n")
print(data.head()) # using .head() function of pandas to quickly look through the dataset

print("\nDescriptive analysis of our Dataset:\n")
print(data.describe(include= 'all'))    # using .describe() function of pandas to calculate the descriptive statistics of our dataset.


# iloc property in pandas gets, or sets, the value(s) of the specified indexes.

# x = np.array(data.iloc[:, np.r_[3:19]]) # input parameters without time data
# x = np.array(data.iloc[:, np.r_[1:4, 12:19]]) # input parameters without crime types
# x = np.array(data.iloc[:, np.r_[0:12, 17:19]]) # input parameters without Location
# x = np.array(data.iloc[:, np.r_[0:18]]) # input parameters without Ins_Out
# x = np.array(data.iloc[:, np.r_[0:19]]) # input parameters with all columns

x = np.array(data.iloc[:, np.r_[0:17, 18:19]]) # input parameters without Street



# the above line of code will get the values of 1st 18 columns from our dataset which are listed below:
#["Time_D","Time_E","Time_M","Type_ARSON","Type_ASSAULT","Type_AUTO","THEFT","Type_BURGLARY","Type_HOMICIDE","Type_LARCENY","Type_RAPE","Type_ROBBERY","Type_SHOOTING","Location_Central","Location_East","Location_North","Location_SouthS","Location_West","Street","Ins_Out"])

y = np.array(data.iloc[:, 19].values) # last column in the dataset (WEAPON)


# creating training and test dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)


# Applying Logistic Regression
log_regression = linear_model.LogisticRegression(max_iter=1500)
log_regression.fit(xtrain, ytrain)

ypred = log_regression.predict(xtest)

# calculating confusion matrix to define the performance of our algorithm
cm = confusion_matrix(ytest, ypred)
print("\nConfusion Matrix: [[True Negatives | False Positives]\n\t\t  [False Negatives | True Positives]]: ",cm)

score = log_regression.score(xtest,ytest)
rounded = round(score*100, 2)
print ('\nModel Score using Logistic Regression: ' + str(rounded) + ' %')
