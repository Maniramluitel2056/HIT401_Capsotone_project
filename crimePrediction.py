import pandas as pd


# loading dataset1 using pandas
df = pd.read_csv("dataset.csv")
print(df.info())


#extracting the rows with weapon value of either firearm or knife
array = ['FIREARM', 'KNIFE'] 
df1 = df.loc[df['Weapon'].isin(array)]
print(df1.head())

# df1.to_csv('extractedDf.csv', index = False)

##########################################################################################
# Performing one hot encoding on column weapon
extractedDf = pd.read_csv("extractedDf.csv")
print(extractedDf.head())

y = pd.get_dummies(extractedDf.Weapon, prefix='Weapon')
# print(y.tail())
# print (y.info())


#######################################################################################
# performing one hot encoding in column premises

# extracting crimes happened on the street from premise column using mapping values technique

streetOffstreet= {  
         "STREET" : "STREET",
         "Street" : "STREET", 
        }

extractedDf["STREET/OFF STREET"] = extractedDf["Premise"].map(streetOffstreet)
extractedDf['STREET/OFF STREET'] = extractedDf['STREET/OFF STREET'].fillna('OFF STREET')

print(extractedDf.tail())
# print(extractedDf.info())

###############################################################################
#extracting and classifying the crime whether the crime has happened outside/inside the premises

inOut= {  
         "ALLEY" : "OUT",
         "APARTMENT" : "IN", 
         "ROW/TOWNHO" : "IN",
        #  "Street" : "STREET",
        #  "STREET" : "STREET",
        #  "Street" : "STREET",
        #  "STREET" : "STREET",
        #  "Street" : "STREET",
        #  "STREET" : "STREET",
        #  "Street" : "STREET",
        #  "STREET" : "STREET",
        #  "Street" : "STREET",
        #  "STREET" : "STREET",
        #  "Street" : "STREET",
        }

extractedDf["IN/OUT"] = extractedDf["Premise"].map(inOut)
extractedDf['IN/OUT'] = extractedDf['IN/OUT'].fillna('OUT')


# extractedDf.to_csv('Df.csv', index = False)
