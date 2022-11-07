#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


#READING THE CSV DATA
data = pd.read_csv('2 months of deliverys 2013.csv', parse_dates=[3], dayfirst = True, header=None)


data = data.rename({0:'Container ID',1:'Container Type',2:'Delivery Type',3:'Date of Availability',4:'Container Origin (Port Location)',5:'EW-Coordinate of Port',6:'NS-Coordinate of Port',7:'Customer Location',8:'EW-Coordinate of Customer', 9:'NS-Coordinate of Customer',10:'Container Weight'},axis=1)


#REMOVING THE REDUNDANT FEATURES
newd = data.drop(['Container ID','NS-Coordinate of Port','EW-Coordinate of Port','NS-Coordinate of Customer','EW-Coordinate of Customer','Container Weight','Customer Location'], axis=1)


#Setting the date column as the index 
newd.set_index('Date of Availability', inplace=True)

Value_counts = []
for i in range(len(newd)):
    Value_counts.append(1)

newd['Values'] = Value_counts

#Creating data for training and testing
new = pd.pivot_table(newd,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum )

new = new.fillna(0)
new = new.resample('1H').sum()


#CHECKING THE SPREAD 
new.boxplot(grid =False)

#Splitting the data in training and testing
X_train, X_test= train_test_split(new, test_size=0.1, shuffle = False)


#Scaling the data for training and testing to accomodate for outliers based on the data
scaler = RobustScaler()
scaler.fit(X_train)
Xscaled = scaler.transform(X_train)
Yscaled = Xscaled[:,:]


#Creating a function to structure the data for the LSTM model 
timesteps = 2
n_inputs = Xscaled.shape[1]

def generator(data,data2,timestamp, n_inputs):
    global genr
    genr = TimeseriesGenerator(data, data2, length = timestamp, batch_size=1)
    return genr

generator(Xscaled,Yscaled,timesteps,n_inputs)


# # BUILDING THE LSTM TRAINING MODEL
model = Sequential()
model.add(LSTM(64,activation='relu',input_shape=(timesteps, n_inputs),return_sequences=False))
model.add(Dense(Yscaled.shape[1]))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


#TRAINING THE MODEL 
model.fit(genr,epochs=50, batch_size=16)
data_loss2 = model.history.history['loss']


#Evaluation of the trained model
plt.plot(range(len(data_loss2)), data_loss2)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Loss Graph')
plt.show()

#testing the performance of the model
test_predictions2 = []
first_batch2 = Xscaled[-timesteps:]
c_batch2 = first_batch2.reshape((1,timesteps, Xscaled.shape[1]))
for i in range(len(X_test)):
    c_pred2 = model.predict(c_batch2)[0]
    test_predictions2.append(c_pred2)
    c_batch2 = np.append(c_batch2[:,1:,:],[[c_pred2]], axis=1)

true_pred2 = scaler.inverse_transform(test_predictions2)

predictions = pd.DataFrame(true_pred2)
predictions= predictions.astype(int)

predictions= predictions.rename({0:('Export', '20DV_y'),1:('Export', '20FL_y'),2:('Export', '20OT_y'),3:('Export', '20RE_y'),4:('Export', '40DV_y'),5:('Export', '40FL_y'),6:('Export', '40HC_y'),7:('Export', '40HR_y'),8:('Export', '40OT_y'),9:('Import', '20DV_y'),10:('Import', '20FL_y'),11:('Import', '20OT_y'),12:('Import', '20RE_y'),13:('Import', '40DV_y'),14:('Import', '40FL_y'),15:('Import', '40HC_y'),16:('Import', '40HP_y'),17:('Import', '40HR_y'),18:('Import', '40OT_y')}, axis = 1)

#creating the data for each location to predict values 
Felixstowe = newd[newd['Container Origin (Port Location)']== 'Felixstowe']
Liverpool = newd[newd['Container Origin (Port Location)']== 'Liverpool']
Felixstowe_Quay = newd[newd['Container Origin (Port Location)']== 'Felixstowe Quay']
Bristol = newd[newd['Container Origin (Port Location)']== 'Bristol']
Teesport = newd[newd['Container Origin (Port Location)']== 'Teesport']
Southampton = newd[newd['Container Origin (Port Location)']== 'Southampton']
Pentalver_Felixstowe = newd[newd['Container Origin (Port Location)']== 'Pentalver Felixstowe']
Hams_Hall = newd[newd['Container Origin (Port Location)']== 'Hams Hall']
Tilbury = newd[newd['Container Origin (Port Location)']== 'Tilbury']
Grangemouth = newd[newd['Container Origin (Port Location)']== 'Grangemouth']
Greenock = newd[newd['Container Origin (Port Location)']== 'Greenock']
Goldstar_Woolpit = newd[newd['Container Origin (Port Location)']== 'Goldstar Woolpit']
OConnors_Transport = newd[newd['Container Origin (Port Location)']== 'OConnors Transport']
Potters_Selby = newd[newd['Container Origin (Port Location)']== 'Potters Selby']


Felixstowe = pd.pivot_table(Felixstowe,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Liverpool = pd.pivot_table(Liverpool,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Felixstowe_Quay = pd.pivot_table(Felixstowe_Quay,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Bristol = pd.pivot_table(Bristol,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Teesport = pd.pivot_table(Teesport,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Southampton = pd.pivot_table(Southampton,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Pentalver_Felixstowe = pd.pivot_table(Pentalver_Felixstowe,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Hams_Hall = pd.pivot_table(Hams_Hall,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Tilbury = pd.pivot_table(Tilbury,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Grangemouth = pd.pivot_table(Grangemouth,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Greenock = pd.pivot_table(Greenock,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Goldstar_Woolpit = pd.pivot_table(Goldstar_Woolpit,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
OConnors_Transport = pd.pivot_table(OConnors_Transport,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)
Potters_Selby = pd.pivot_table(Potters_Selby,values = 'Values', index = ['Date of Availability'],columns = ['Delivery Type','Container Type'],aggfunc=np.sum, fill_value=0)


Felixstowe = Felixstowe.resample('1H').sum()
Liverpool = Liverpool.resample('1H').sum()
Felixstowe_Quay = Felixstowe_Quay.resample('1H').sum()
Bristol = Bristol.resample('1H').sum()
Teesport = Teesport.resample('1H').sum()
Southampton = Southampton.resample('0.5H').sum()
Pentalver_Felixstowe = Pentalver_Felixstowe.resample('0.5H').sum()
Hams_Hall = Hams_Hall.resample('0.5H').sum()
Tilbury = Tilbury.resample('0.5H').sum()
Grangemouth = Grangemouth.resample('0.5H').sum()
Greenock = Greenock.resample('1H').sum()
Goldstar_Woolpit = Goldstar_Woolpit.resample('0.5H').sum()
OConnors_Transport = OConnors_Transport.resample('2H').sum()
Potters_Selby =Potters_Selby.resample('1H').sum()


#Adding the missing columns to resemble the data
Felixstowe[('Import', '40HP')] = 0
Liverpool[('Export','20FL')],Liverpool[('Export','40FL')], Liverpool[('Import','20FL')],Liverpool[('Import','40FL')],Liverpool[('Import','40HP')],Liverpool[('Import','40OT')]= 0,0,0,0,0,0
Felixstowe_Quay[('Export', '20DV')],Felixstowe_Quay[('Export', '20FL')],Felixstowe_Quay[('Export', '20OT')],Felixstowe_Quay[('Export', '20RE')],Felixstowe_Quay[('Export', '40DV')],Felixstowe_Quay[('Export', '40FL')],Felixstowe_Quay[('Export', '40HC')],Felixstowe_Quay[('Export', '40HR')],Felixstowe_Quay[('Export', '40OT')],Felixstowe_Quay[('Import', '20RE')],Felixstowe_Quay[('Import', '40FL')]=0,0,0,0,0,0,0,0,0,0,0
Bristol[('Export', '20FL')],Bristol[('Export', '20RE')],Bristol[('Export', '40FL')],Bristol[('Export', '40OT')],Bristol[('Import', '20FL')],Bristol[('Import', '20OT')],Bristol[('Import', '20RE')],Bristol[('Import', '40FL')],Bristol[('Import', '40HP')],Bristol[('Import', '40HR')],Bristol[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0
Teesport[('Export', '20FL')],Teesport[('Export', '20RE')],Teesport[('Export', '40FL')],Teesport[('Export', '40HR')],Teesport[('Import', '20FL')],Teesport[('Import', '20OT')],Teesport[('Import', '20RE')],Teesport[('Import', '40FL')],Teesport[('Import', '40HP')],Teesport[('Import', '40HR')]=0,0,0,0,0,0,0,0,0,0
Southampton[('Export', '20FL')],Southampton[('Export', '20OT')],Southampton[('Export', '40FL')],Southampton[('Export', '40HR')],Southampton[('Import', '20FL')],Southampton[('Import', '20RE')],Southampton[('Import', '40FL')],Southampton[('Import', '40HP')],Southampton[('Import', '40HR')],Southampton[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0
Pentalver_Felixstowe[('Export', '20DV')],Pentalver_Felixstowe[('Export', '20FL')],Pentalver_Felixstowe[('Export', '20OT')],Pentalver_Felixstowe[('Export', '20RE')],Pentalver_Felixstowe[('Export', '40DV')],Pentalver_Felixstowe[('Export', '40FL')],Pentalver_Felixstowe[('Export', '40HC')],Pentalver_Felixstowe[('Export', '40HR')],Pentalver_Felixstowe[('Export', '40OT')],Pentalver_Felixstowe[('Import', '20FL')],Pentalver_Felixstowe[('Import', '20RE')],Pentalver_Felixstowe[('Import', '40FL')],Pentalver_Felixstowe[('Import', '40HP')],Pentalver_Felixstowe[('Import', '40HR')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Hams_Hall[('Export','20FL')],Hams_Hall[('Export','20OT')],Hams_Hall[('Export','20RE')],Hams_Hall[('Export','40FL')],Hams_Hall[('Export','40HR')],Hams_Hall[('Export','40OT')],Hams_Hall[('Import','20FL')],Hams_Hall[('Import','20OT')],Hams_Hall[('Import','20RE')],Hams_Hall[('Import','40FL')],Hams_Hall[('Import','40HP')],Hams_Hall[('Import','40HR')],Hams_Hall[('Import','40OT')]=0,0,0,0,0,0,0,0,0,0,0,0,0
Tilbury[('Export', '40FL')],Tilbury[('Export', '40OT')],Tilbury[('Import', '20FL')],Tilbury[('Import', '40FL')],Tilbury[('Import', '40HP')],Tilbury[('Import', '40OT')] = 0,0,0,0,0,0
Grangemouth[('Export', '20FL')],Grangemouth[('Export', '40FL')],Grangemouth[('Import', '20RE')],Grangemouth[('Import', '40HP')],Grangemouth[('Import', '40HR')]=0,0,0,0,0
Greenock[('Export','20FL')],Greenock[('Export','20OT')],Greenock[('Export','20RE')],Greenock[('Export','40FL')],Greenock[('Export','40HR')],Greenock[('Export','40OT')],Greenock[('Import','20FL')],Greenock[('Import','20OT')],Greenock[('Import','20RE')],Greenock[('Import','40DV')],Greenock[('Import','40FL')],Greenock[('Import','40HP')],Greenock[('Import','40HR')],Greenock[('Import','40OT')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Goldstar_Woolpit[('Export', '20DV')],Goldstar_Woolpit[('Export', '20FL')],Goldstar_Woolpit[('Export', '20OT')],Goldstar_Woolpit[('Export', '20RE')],Goldstar_Woolpit[('Export', '40DV')],Goldstar_Woolpit[('Export', '40FL')],Goldstar_Woolpit[('Export', '40HC')],Goldstar_Woolpit[('Export', '40HR')], Goldstar_Woolpit[('Export', '40OT')],Goldstar_Woolpit[('Import', '20FL')],Goldstar_Woolpit[('Import', '20OT')],Goldstar_Woolpit[('Import', '20RE')],Goldstar_Woolpit[('Import', '40FL')],Goldstar_Woolpit[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0,0,0,0
OConnors_Transport[('Export', '20DV')],OConnors_Transport[('Export', '20FL')],OConnors_Transport[('Export', '20OT')],OConnors_Transport[('Export', '20RE')],OConnors_Transport[('Export', '40FL')],OConnors_Transport[('Export', '40HR')],OConnors_Transport[('Export', '40OT')],OConnors_Transport[('Import', '20FL')],OConnors_Transport[('Import', '20OT')],OConnors_Transport[('Import', '20RE')],OConnors_Transport[('Import', '40FL')],OConnors_Transport[('Import', '40HP')],OConnors_Transport[('Import', '40HR')],OConnors_Transport[('Import', '40OT')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Potters_Selby[('Export', '20FL')],Potters_Selby[('Export', '20OT')],Potters_Selby[('Export', '20RE')],Potters_Selby[('Export', '40FL')],Potters_Selby[('Export', '40HR')],Potters_Selby[('Export', '40OT')],Potters_Selby[('Import', '20FL')],Potters_Selby[('Import', '20RE')],Potters_Selby[('Import', '40FL')],Potters_Selby[('Import', '40HP')],Potters_Selby[('Import', '40HR')],Potters_Selby[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0,0


#Rearranging the data to resemble the model
Felixstowe = Felixstowe[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Liverpool = Liverpool[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Felixstowe_Quay = Felixstowe_Quay[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Bristol = Bristol[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Teesport = Teesport[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Southampton = Southampton[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Pentalver_Felixstowe = Pentalver_Felixstowe[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Hams_Hall = Hams_Hall[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Tilbury = Tilbury[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Grangemouth = Grangemouth[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Greenock = Greenock[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Goldstar_Woolpit = Goldstar_Woolpit[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
OConnors_Transport = OConnors_Transport[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]
Potters_Selby = Potters_Selby[[('Export', '20DV'),('Export', '20FL'),('Export', '20OT'),('Export', '20RE'),('Export', '40DV'),('Export', '40FL'),('Export', '40HC'),('Export', '40HR'),('Export', '40OT'),('Import', '20DV'),('Import', '20FL'),('Import', '20OT'),('Import', '20RE'),('Import', '40DV'),('Import', '40FL'),('Import', '40HC'),('Import', '40HP'),('Import', '40HR'),('Import', '40OT')]]


scaler2 = RobustScaler()
scaler3 = RobustScaler()
scaler4 = RobustScaler()
scaler5 = RobustScaler()
scaler6 = RobustScaler()
scaler7 = RobustScaler()
scaler8 = RobustScaler()
scaler9 = RobustScaler()
scaler10 = RobustScaler()
scaler11 = RobustScaler()
scaler12 = RobustScaler()
scaler13 = RobustScaler()
scaler14 = RobustScaler()
scaler15 = RobustScaler()

#SCALING THE PREDICTION DATA
scaler2.fit(Felixstowe)
FelixstoweS = scaler2.transform(Felixstowe)
scaler3.fit(Liverpool)
LiverpoolS = scaler3.transform(Liverpool)
scaler4.fit(Felixstowe_Quay)
Felixstowe_QuayS = scaler4.transform(Felixstowe_Quay)
scaler5.fit(Bristol)
BristolS = scaler5.transform(Bristol)
scaler6.fit(Teesport)
TeesportS = scaler6.transform(Teesport)
scaler7.fit(Southampton)
SouthamptonS = scaler7.transform(Southampton)
scaler8.fit(Pentalver_Felixstowe)
Pentalver_FelixstoweS = scaler8.transform(Pentalver_Felixstowe)
scaler9.fit(Hams_Hall)
Hams_HallS = scaler9.transform(Hams_Hall)
scaler10.fit(Tilbury)
TilburyS = scaler10.transform(Tilbury)
scaler11.fit(Grangemouth)
GrangemouthS = scaler11.transform(Grangemouth)
scaler12.fit(Greenock)
GreenockS = scaler12.transform(Greenock)
scaler13.fit(Goldstar_Woolpit)
Goldstar_WoolpitS = scaler13.transform(Goldstar_Woolpit)
scaler14.fit(OConnors_Transport)
OConnors_TransportS = scaler14.transform(OConnors_Transport)
scaler15.fit(Potters_Selby)
Potters_SelbyS = scaler15.transform(Potters_Selby)

#Creating a date range for the predicted data for 7 days
future = 672
FelixstoweD = pd.date_range(start = Felixstowe.index[-1], periods = 24, freq = '1H').to_list()
LiverpoolD = pd.date_range(start = Liverpool.index[-1], periods = 24, freq = '1H').to_list()
Felixstowe_QuayD = pd.date_range(start = Felixstowe_Quay.index[-1], periods = 24, freq = '1H').to_list()
BristolD = pd.date_range(start = Bristol.index[-1], periods = 24, freq = '1H').to_list()
TeesportD = pd.date_range(start = Teesport.index[-1], periods = 24, freq = '1H').to_list()
SouthamptonD = pd.date_range(start = Southampton.index[-1], periods = 48, freq = '0.5H').to_list()
Pentalver_FelixstoweD = pd.date_range(start = Pentalver_Felixstowe.index[-1], periods = 48, freq = '0.5H').to_list()
Hams_HallD = pd.date_range(start = Hams_Hall.index[-1], periods = 192, freq = '0.5H').to_list()
TilburyD = pd.date_range(start = Tilbury.index[-1], periods = 48, freq = '0.5H').to_list()
GrangemouthD = pd.date_range(start = Grangemouth.index[-1], periods = 48, freq = '0.25H').to_list()
GreenockD = pd.date_range(start = Greenock.index[-1], periods = 24, freq = '1H').to_list()
Goldstar_WoolpitD = pd.date_range(start =Goldstar_Woolpit.index[-1], periods = future, freq = '0.25H').to_list()
OConnors_TransportD = pd.date_range(start = OConnors_Transport.index[-1], periods = 12, freq = '2H').to_list()
Potters_SelbyD = pd.date_range(start = Potters_Selby.index[-1], periods = 24, freq = '1H').to_list()

FelixstoweDates = []
LiverpoolDates = []
Felixstowe_QuayDates = []
BristolDates = []
TeesportDates = []
SouthamptonDates = []
Pentalver_FelixstoweDates = []
Hams_HallDates = []
TilburyDates = []
GrangemouthDates = []
GreenockDates = []
Goldstar_WoolpitDates = []
OConnors_TransportDates = []
Potters_SelbyDates = []

for i in FelixstoweD:
    FelixstoweDates.append(i)
for j in LiverpoolD:
    LiverpoolDates.append(j)
for k in Felixstowe_QuayD:
    Felixstowe_QuayDates.append(k)
for l in BristolD:
    BristolDates.append(l)
for m in TeesportD:
    TeesportDates.append(m)
for n in SouthamptonD:
    SouthamptonDates.append(n)
for o in Pentalver_FelixstoweD:
    Pentalver_FelixstoweDates.append(o)
for p in Hams_HallD:
    Hams_HallDates.append(p)
for q in TilburyD:
    TilburyDates.append(q)
for r in GrangemouthD:
    GrangemouthDates.append(r)
for s in GreenockD:
    GreenockDates.append(s)
for t in Goldstar_WoolpitD:
    Goldstar_WoolpitDates.append(t)
for u in OConnors_TransportD:
    OConnors_TransportDates.append(u)
for v in Potters_SelbyD:
    Potters_SelbyDates.append(v)


Potters_Selby[('Export', '20FL')],Potters_Selby[('Export', '20OT')],Potters_Selby[('Export', '20RE')],Potters_Selby[('Export', '40FL')],Potters_Selby[('Export', '40HR')],Potters_Selby[('Export', '40OT')],Potters_Selby[('Import', '20FL')],Potters_Selby[('Import', '20RE')],Potters_Selby[('Import', '40FL')],Potters_Selby[('Import', '40HP')],Potters_Selby[('Import', '40HR')],Potters_Selby[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0,0



OConnors_Transport[('Export', '20DV')],OConnors_Transport[('Export', '20FL')],OConnors_Transport[('Export', '20OT')],OConnors_Transport[('Export', '20RE')],OConnors_Transport[('Export', '40FL')],OConnors_Transport[('Export', '40HR')],OConnors_Transport[('Export', '40OT')],OConnors_Transport[('Import', '20FL')],OConnors_Transport[('Import', '20OT')],OConnors_Transport[('Import', '20RE')],OConnors_Transport[('Import', '40FL')],OConnors_Transport[('Import', '40HP')],OConnors_Transport[('Import', '40HR')],OConnors_Transport[('Import', '40OT')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Goldstar_Woolpit[('Export', '20DV')],Goldstar_Woolpit[('Export', '20FL')],Goldstar_Woolpit[('Export', '20OT')],Goldstar_Woolpit[('Export', '20RE')],Goldstar_Woolpit[('Export', '40DV')],Goldstar_Woolpit[('Export', '40FL')],Goldstar_Woolpit[('Export', '40HC')],Goldstar_Woolpit[('Export', '40HR')], Goldstar_Woolpit[('Export', '40OT')],Goldstar_Woolpit[('Import', '20FL')],Goldstar_Woolpit[('Import', '20OT')],Goldstar_Woolpit[('Import', '20RE')],Goldstar_Woolpit[('Import', '40FL')],Goldstar_Woolpit[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0,0,0,0
Greenock[('Export','20FL')],Greenock[('Export','20OT')],Greenock[('Export','20RE')],Greenock[('Export','40FL')],Greenock[('Export','40HR')],Greenock[('Export','40OT')],Greenock[('Import','20FL')],Greenock[('Import','20OT')],Greenock[('Import','20RE')],Greenock[('Import','40DV')],Greenock[('Import','40FL')],Greenock[('Import','40HP')],Greenock[('Import','40HR')],Greenock[('Import','40OT')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Grangemouth[('Export', '20FL')],Grangemouth[('Export', '40FL')],Grangemouth[('Import', '20RE')],Grangemouth[('Import', '40HP')],Grangemouth[('Import', '40HR')]=0,0,0,0,0
Tilbury[('Export', '40FL')],Tilbury[('Export', '40OT')],Tilbury[('Import', '20FL')],Tilbury[('Import', '40FL')],Tilbury[('Import', '40HP')],Tilbury[('Import', '40OT')] = 0,0,0,0,0,0
Hams_Hall[('Export','20FL')],Hams_Hall[('Export','20OT')],Hams_Hall[('Export','20RE')],Hams_Hall[('Export','40FL')],Hams_Hall[('Export','40HR')],Hams_Hall[('Export','40OT')],Hams_Hall[('Import','20FL')],Hams_Hall[('Import','20OT')],Hams_Hall[('Import','20RE')],Hams_Hall[('Import','40FL')],Hams_Hall[('Import','40HP')],Hams_Hall[('Import','20HR')],Hams_Hall[('Import','40OT')]=0,0,0,0,0,0,0,0,0,0,0,0,0
Pentalver_Felixstowe[('Export', '20DV')],Pentalver_Felixstowe[('Export', '20FL')],Pentalver_Felixstowe[('Export', '20OT')],Pentalver_Felixstowe[('Export', '20RE')],Pentalver_Felixstowe[('Export', '40DV')],Pentalver_Felixstowe[('Export', '40FL')],Pentalver_Felixstowe[('Export', '40HC')],Pentalver_Felixstowe[('Export', '240HR')],Pentalver_Felixstowe[('Export', '40OT')],Pentalver_Felixstowe[('Import', '20FL')],Pentalver_Felixstowe[('Import', '20RE')],Pentalver_Felixstowe[('Import', '40FL')],Pentalver_Felixstowe[('Import', '40HP')],Pentalver_Felixstowe[('Import', '40HR')] = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
Southampton[('Export', '20FL')],Southampton[('Export', '20OT')],Southampton[('Export', '40FL')],Southampton[('Export', '40HR')],Southampton[('Import', '20FL')],Southampton[('Import', '20RE')],Southampton[('Import', '40FL')],Southampton[('Import', '40HP')],Southampton[('Import', '40HR')],Southampton[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0
Teesport[('Export', '20FL')],Teesport[('Export', '20RE')],Teesport[('Export', '40FL')],Teesport[('Export', '40HR')],Teesport[('Import', '20FL')],Teesport[('Import', '20OT')],Teesport[('Import', '20RE')],Teesport[('Import', '40FL')],Teesport[('Import', '40HP')],Teesport[('Import', '40HR')]=0,0,0,0,0,0,0,0,0,0
Bristol[('Export', '20FL')],Bristol[('Export', '20RE')],Bristol[('Export', '40FL')],Bristol[('Export', '40OT')],Bristol[('Import', '20FL')],Bristol[('Import', '20OT')],Bristol[('Import', '20RE')],Bristol[('Import', '40FL')],Bristol[('Import', '40HP')],Bristol[('Import', '40HR')],Bristol[('Import', '40OT')]=0,0,0,0,0,0,0,0,0,0,0
Felixstowe_Quay[('Export', '20DV')],Felixstowe_Quay[('Export', '20FL')],Felixstowe_Quay[('Export', '20OT')],Felixstowe_Quay[('Export', '20RE')],Felixstowe_Quay[('Export', '40DV')],Felixstowe_Quay[('Export', '40FL')],Felixstowe_Quay[('Export', '40HC')],Felixstowe_Quay[('Export', '40HR')],Felixstowe_Quay[('Export', '400T')],Felixstowe_Quay[('Import', '20RE')],Felixstowe_Quay[('Import', '40FL')]=0,0,0,0,0,0,0,0,0,0,0
Liverpool[('Export','20FL')],Liverpool[('Export','40FL')], Liverpool[('Import','20FL')],Liverpool[('Import','40FL')],Liverpool[('Import','40HP')],Liverpool[('Import','40OT')]= 0,0,0,0,0,0

# # PREDICTING DEMAND AND SUPPLY FOR EACH LOCATION
#Felixstowe Predictions
test_predictions1 = []
first_batch1 = FelixstoweS[-timesteps:]
c_batch1 = first_batch1.reshape((1,timesteps, FelixstoweS.shape[1]))
for i in range(len(FelixstoweDates)):
    c_pred1 = model.predict(c_batch1)[0]
    test_predictions1.append(c_pred1)
    c_batch1 = np.append(c_batch1[:,1:,:],[[c_pred1]], axis=1)

true_p1 = scaler2.inverse_transform(test_predictions1)
FelixstoweP = true_p1.astype(int)
FelixstoweP = pd.DataFrame(FelixstoweP)
FDates = pd.DataFrame(FelixstoweDates)

FelixstoweP = FelixstoweP.rename({0:('Export', '20DV_y'),1:('Export', '20FL_y'),2:('Export', '20OT_y'),3:('Export', '20RE_y'),4:('Export', '40DV_y'),5:('Export', '40FL_y'),6:('Export', '40HC_y'),7:('Export', '40HR_y'),8:('Export', '40OT_y'),9:('Import', '20DV_y'),10:('Import', '20FL_y'),11:('Import', '20OT_y'),12:('Import', '20RE_y'),13:('Import', '40DV_y'),14:('Import', '40FL_y'),15:('Import', '40HC_y'),16:('Import', '40HP_y'),17:('Import', '40HR_y'),18:('Import', '40OT_y')}, axis = 1)
FelixstoweP['Dates'] = FDates
FelixstoweP = FelixstoweP.set_index('Dates')
FelixstoweP



#Liverpool Predictions
test_predictions2 = []
first_batch2 = LiverpoolS[-timesteps:]
c_batch2 = first_batch2.reshape((1,timesteps, LiverpoolS.shape[1]))
for i in range(len(LiverpoolDates)):
    c_pred2 = model.predict(c_batch2)[0]
    test_predictions2.append(c_pred2)
    c_batch2 = np.append(c_batch2[:,1:,:],[[c_pred2]], axis=1)

true_p2 = scaler2.inverse_transform(test_predictions2)
LiverpoolP = true_p2.astype(int)
LiverpoolP = pd.DataFrame(LiverpoolP)
LDates = pd.DataFrame(LiverpoolDates)

LiverpoolP = LiverpoolP.rename({0:('Export', '20DV_y'),1:('Export', '20FL_y'),2:('Export', '20OT_y'),3:('Export', '20RE_y'),4:('Export', '40DV_y'),5:('Export', '40FL_y'),6:('Export', '40HC_y'),7:('Export', '40HR_y'),8:('Export', '40OT_y'),9:('Import', '20DV_y'),10:('Import', '20FL_y'),11:('Import', '20OT_y'),12:('Import', '20RE_y'),13:('Import', '40DV_y'),14:('Import', '40FL_y'),15:('Import', '40HC_y'),16:('Import', '40HP_y'),17:('Import', '40HR_y'),18:('Import', '40OT_y')}, axis = 1)
LiverpoolP['Dates']=LDates
LiverpoolP = LiverpoolP.set_index('Dates')







