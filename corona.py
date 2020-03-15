import numpy as np
import pandas as pd
import seaborn as snp
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# corona data fetching
corona_data = pd.read_csv('2019_nCoV_20200121_20200206.csv')
corona_data.head()
corona_data.info()
corona_data.describe().transpose()

# processing missing values
corona_data['Province/State'].fillna('Other', inplace=True)
corona_data.fillna(value=0.0, inplace=True)

# EDA
plt.figure(figsize=(10,7))
snp.pairplot(corona_data)
snp.heatmap(corona_data.corr(), annot=True, cmap='coolwarm')
snp.countplot('Province/State', hue='Death', data=corona_data)
snp.jointplot(x='Confirmed', y='Recovered', data=corona_data)
snp.scatterplot(x='Confirmed', y='Recovered', hue='Country/Region',data=corona_data, legend=False)

# add day-month-year column
corona_data['Last Update'] = pd.to_datetime(corona_data['Last Update'])
corona_data['month'] = pd.DatetimeIndex(corona_data['Last Update']).month
corona_data['year'] = pd.DatetimeIndex(corona_data['Last Update']).year
corona_data['day'] = pd.DatetimeIndex(corona_data['Last Update']).day
corona_data['time'] = pd.DatetimeIndex(corona_data['Last Update']).time
corona_data.drop('Last Update', axis=1, inplace=True)

# train and test data splitting
X = corona_data.drop(['Province/State','Country/Region', 'time', 'Death'], axis=1)
y = corona_data['Death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)
reg_model_pred = reg.predict(X_test) 

# evaluation of model-linear regression
mean_absolute_error(y_test, reg_model_pred)
mean_squared_error(y_test, reg_model_pred)
np.sqrt(mean_squared_error(y_test, reg_model_pred))
snp.scatterplot(x=y_test, y=reg_model_pred)

# new data
new_corona_data = corona_data.drop(['Province/State','Country/Region', 'time'], axis=1)

# scaled data transformation
scaled = StandardScaler()
scaled.fit(new_corona_data)
scaled_data = scaled.transform(new_corona_data)

new_corona  = pd.DataFrame(data = scaled_data, columns=new_corona_data.columns)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(new_corona.drop('Death', axis=1), new_corona['Death'], test_size=0.33, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)

# model evaluation
mean_absolute_error(y_test, rf_pred)
mean_squared_error(y_test, rf_pred)
np.sqrt(mean_squared_error(y_test, rf_pred))
snp.lmplot(x='Death', y='predicted_value', data=df)

# artificial neural networks
model = Sequential()
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train.values, y=y_train.values, validation_data=(X_test, y_test), epochs=450, verbose=1)
loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
predict = model.predict(X_test.values)

# model evaluation
mean_absolute_error(y_test, predict)
mean_squared_error(y_test,predict)

# model prediction on new data
new_data = new_corona.drop('Death', axis=1).iloc[3]
model.predict(new_data.values.reshape(-1,6))
new_corona.iloc[3]

plt.show()