import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("CrimaPreProcessed.csv", encoding= 'latin-1')
print(data.info())

model = BernoulliNB(binarize=0.0)

def modelGen(x,y):
   xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)
   model.fit(xtrain, ytrain)
   return model.score(xtest, ytest)*100

#Target
y = np.array(data.iloc[:, 19].values)#feature ["Weapon"]

#Model 1
#Input feature
x = np.array(data.iloc[:,np.r_[0:19]].values)#Input features all columns: ["Time_D","Time_E","Time_M","Type_ARSON","Type_ASSAULT","Type_AUTO","THEFT","Type_BURGLARY","Type_HOMICIDE","Type_LARCENY","Type_RAPE","Type_ROBBERY","Type_SHOOTING","Location_Central","Location_East","Location_North","Location_SouthS","Location_West","Street","Ins_Out"])
print("\n Model 1 Input Features are ", data.columns[np.r_[0:19]] ,"\n Model 1 Accuracy is %0.2f " % modelGen(x,y))

#Model 2 
#input feature
x = np.array(data.iloc[:,np.r_[0:17, 18:19]].values)#Excluded feature ["Street"])
print("\n Model 2 Input Features are ", data.columns[np.r_[0:17, 18:19]] ,"\n Model 2 Accuracy is  %0.2f" % modelGen(x,y))

#Model 3
#input features
x = np.array(data.iloc[:, np.r_[3:19]]) # input parameters without time data
print("\n Model 3 Input Features are ", data.columns[np.r_[3:19]] ,"\n Model 3 Accuracy is  %0.2f" % modelGen(x,y))

#Model 4
#input features
x = np.array(data.iloc[:, np.r_[1:4, 12:19]]) # input parameters without Crime types
print("\n Model 4 Input Features are ", data.columns[np.r_[3:19]] ,"\n Model 4 Accuracy is  %0.2f" % modelGen(x,y))

# Model 5
x = np.array(data.iloc[:, np.r_[0:12, 17:19]]) # input parameters without Location
print("\n Model 5 Input Features are ", data.columns[np.r_[0:12, 17:19]] ,"\n Model 5 Accuracy is  %0.2f" % modelGen(x,y))

# Model 6
x = np.array(data.iloc[:, np.r_[0:18]]) # input parameters without Ins_Out
print("\n Model 6 Input Features are ", data.columns[np.r_[0:18]] ,"\n Model 6 Accuracy is  %0.2f" % modelGen(x,y))




   


