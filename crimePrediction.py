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
          "APT.LOCKE" : "IN",
         "APT/CONDO" : "IN",
         "ATM MACHIN" : "IN",
         "AUTO PARTS" : "IN",
         "BAKERY" : "IN",
         "BANK/FINAN" : "IN",
         "BAR" : "IN",
         "BARBER/BEA" : "IN",
         "BRIDGE-PIE" : "OUT",
         "BUS. STOR" : "OUT",
         "BUS. PARK" : "OUT",
         "BUS/RAILRO" : "OUT"
         "BUS/AUTO" : "OUT"
         "CAB" : "OUT",
         "CAR REPAI" : "OUT"
         "CAR LOT-NE" : "OUT"
         "CARRY OUT" : "OUT"
         "CEMETERY" : "OUT",
         "CHAIN FOOD" : "IN",
         "CHURCH" : "IN",
         "CLOTHING" : "IN",
         "CLUB HOUSE" : "IN",
         "COMMON BUS" : "OUT",
         "CONSTRUCTI" : "OUT",
         "CONVENIENC" : "OUT",
         "CONVENTION" : "OUT",
         "COURT HOUS" : "IN",
         "DEPARTMENT" : "IN", 
         "DRIVEWAY" : "OUT",
         "DRUNG STORE" : "IN",
         "DWELLING" : "OUT",
         "FAST FOOD" : "IN",
         "FINANCE/LO" : "IN",
         "FIRE DEPAR" : "IN",
         "GARAGE" : "OUT",
         "GARAGE ON" : "OUT",
         "GAS STATIO" : "OUT",
         "GROCERY/CO" : "IN",
         "HARDWARE/B" : "IN",
         "HOSP/NURS." : "IN"
         "HOSPITAL" : "IN"
         "HOTEL/MOTE" : "IN",
         "INNER HARB" : "OUT"
         "JEWELRY ST" : "IN"
         "LAUNDRY/CL" : "IN"
         "LIBRARY" : "IN",
         "LIGHT RAIL" : "OUT",
         "LIQUOR STO" : "OUT",
         "MARKET STA" : "IN",
         "MTA LOT" : "IN",
         "NIGHT DEPO" : "OUT",
         "OFFICE BUI" : "IN",
         "OTHER - IN" : "IN",
         "OTHER - OU" : "OUT",
         "OTHER/RESI" : "IN",
         "PARK" : "OUT",
         "PARKING LO" : "OUT",
         "PIZZA/OTHE" : "IN",
         "PLAYGROUND" : "OUT",
         "POLICE DEP" : "IN",
         "POOL/BOWLI" : "OUT",
         "PORCH/DECK : "OUT",
         "Private Sc" : "IN",
         "Public Are" : "OUT",
         "PUBLIC BUI" : "OUT",
         "PUBLIC HOU" : "OUT",
         "Public Sch" : "IN",
         "RACE TRACK" : "OUT"
         "RAILROAD C" : "OUT"
         "RECREATION" : "IN",
         "RELIGIOUS" : "OUT"
         "RENTAL/VID" : "OUT"
         "RESTAURANT" : "OUT"
         "RETAIL/SMA" : "IN",
         "ROW/TOWNHO" : "IN",
         "SALESMAN/C" : "IN",
         "SCHOOL" : "IN",
         "SCHOOL PLA" : "IN",
         "SHED/GARAG" : "IN",
         "SHOPPING M" : "IN",
         "SINGLE HOU" : "OUT",
         "SKYWALK" : "OUT",
         "SPECIALITY" : "IN",
         "STADIUM" : "OUT", 
         "Street" : "OUT",
         "STRUCTURE" : "OUT",
         "SUBWAY" : "OUT",
         "TAVERN/NIG" : "OUT"
         "THEATRE" : "OUT"
         "UNKNOWN" : "OUT",
         "VACANT BUI" : "OUT"
         "Vacant Dwe" : "OUT"
         "VACANT LOT" : "OUT"
         "Vehicle" : "OUT",
         "WAREHOUSE" : "OUT",
         "WHOLESALE/" : "IN",
         "YARD" : "OUT",
         "YARD/BUSIN" : "OUT",
         "" : "STREET",
        }

extractedDf["IN/OUT"] = extractedDf["Premise"].map(inOut)
extractedDf['IN/OUT'] = extractedDf['IN/OUT'].fillna('OUT')


# extractedDf.to_csv('Df.csv', index = False)
