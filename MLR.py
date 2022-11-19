#Evaluation of the WEAPON feature from BPD Crime bpd_crime_dataframeset (pre-processed) using Multinomial Logistic Regression

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



# Load preprocessed bpd_crime_dataframeset
data = pd.read_csv('CrimaPreProcessed.csv')

print("\n\n#####################################################################################")
print("########################### Multinomial Logistic Regression #########################")
print("#####################################################################################")


print("\n\nDataset Info: ")
print(data.info())  #using .info() function of pandas to explore the dataset

print("\nFirst five rows of our Dataset: \n")
print(data.head()) # using .head() function of pandas to quickly look through the dataset

print("\nDescriptive analysis of our Dataset:\n")
print(data.describe(include= 'all'))    # using .describe() function of pandas to calculate the descriptive statistics of our dataset.



def modelGenerate (startColumn1, endColumn1, startColumn2, endColumn2, modelNo):
    print("\n\n################ Model "+str(modelNo)+" ################### ")

    x = np.array(data.iloc[:, np.r_[startColumn1:endColumn1, startColumn2:endColumn2]]) # iloc property in pandas gets, or sets, the value(s) of the specified indexes.
    y = np.array(data.iloc[:, 19].values) # last column in the dataset (WEAPON)


    # creating training and test dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)


    # Applying Logistic Regression
    log_regression = linear_model.LogisticRegression(max_iter=1500)
    log_regression.fit(xtrain, ytrain)

    ypred = log_regression.predict(xtest)


    print("\nInput Features for Model " + str(modelNo) +  " :",data.columns[np.r_[startColumn1:endColumn1, startColumn2:endColumn2]])
    
    
    # calculating confusion matrix to define the performance of our algorithm
    cm = confusion_matrix(ytest, ypred)
    print("\nConfusion Matrix for this model: [[True Negatives | False Positives]\n\t\t  [False Negatives | True Positives]]: ",cm)

    score = log_regression.score(xtest,ytest)
    rounded = round(score*100, 2)
    print ('\nModel ' +str(modelNo) + ' score using Logistic Regression: ' + str(rounded) + ' %.')




# Model generation with distinct input features

# Model 1
modelGenerate(0, 15, 15, 19, 1)

# Model 2
modelGenerate(3, 15, 15, 19, 2)

# Model 3
modelGenerate(1, 4, 12, 19, 3)

# Model 4
modelGenerate(0, 12, 17, 19, 4)

# Model 5
modelGenerate(0, 17, 18, 19, 5)

# Model 6
modelGenerate(0, 17, 17, 18, 6)
