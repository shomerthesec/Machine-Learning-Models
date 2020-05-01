# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:14:53 2020

@author: ShomerTheSec
"""
#%% importing the data 

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import pandas as pd
import quandl
import math
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


#%% modifying the data and taking the intersting arrays
data = quandl.get("WIKI/GOOGL",authtoken="sN8T385aFHpzuC8euzPc")

data = data[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# creating the high - low ratio 

data['HL_PCT'] = (data['Adj. High'] - data['Adj. Low']) / data['Adj. Close'] * 100.0

#creating the percentage of change from opening to closing

data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100.0
data = data[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#%% 

#chosing the label column 
predict_col = 'Adj. Close'

# replacing all the nand values with -99999
data.fillna(value=-99999, inplace=True)

# to predict 0.01 from the training set -for 100 days it will predict1 day-
predict_out = int(math.ceil(0.01 * len(data))) 
data['label'] = data[predict_col].shift(-predict_out)

#%% splitting the data into Features X and labels y 

data.dropna(inplace=True)
X = np.array(data.drop(['label'], 1))
y = np.array(data['label'])
X = preprocessing.scale(X)
y = np.array(data['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#%% training and trying different kernerls for SVM model
for k in ['linear','poly','rbf','sigmoid']:
    model = svm.SVR(kernel=k, gamma='scale')
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(k,accuracy)
  
#%% training linearregression model
model = LinearRegression()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print('accuracy ',accuracy)

#%% chosing the svm linear kernel for best accuracy

model = svm.SVR(kernel='linear', gamma='scale')
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

#%% to make predictions
  
X_lately = X[-predict_out:]

predict_set = model.predict(X_lately)

#%% to prepare for plotting

data['predict'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predict_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]

#%% to plot the graph
data['Adj. Close'].plot()
data['predict'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#%% to save the model
import pickle
with open('linearregression.pickle','wb') as f:
    pickle.dump(model, f)

#%% to load the model
pickle_in = open('linearregression.pickle','rb')
model = pickle.load(pickle_in)