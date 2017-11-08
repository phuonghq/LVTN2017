from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import csv
import datetime 
import os
import tensorflow as tf


#avg
def avg(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
#print(agg)

    return agg
 
# load dataset
rootPath = os.path.abspath(os.path.dirname(__file__)) 
if __name__ == "__main__":
    print("--- Loading Data ---")
    test_size =  10000
    for subDir in os.listdir(rootPath + "/Data"):
        sub_dir = rootPath + "/Data/" + subDir
        if (os.path.isdir(sub_dir)):
            for filePath in os.listdir(sub_dir):
                filePath = sub_dir + "/" + filePath
              #  X, Y, Z = data_loader(filePath)
               # X_train, Y_train, Z_train, X_test, Y_test, Z_test = train_test_split(X, Y, Z, test_size)
                #print("--- Printed ---")
               # print(type(X_train)) #time
                #print(X_train[0])
               # print(Y_train) #segment
              #  print(Z_train) #velocity
dataset = read_csv('2017-05-01.csv', header=0, index_col=0)
values = dataset.values
#print(X_train.shape)
print(values)
#print(values[:][1:])

#for i in range(len(values) - 1):
 #   values[i] = numpy.delete(values[i], 0)

#values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 60))
#scaled = scaler.fit_transform(values[:][1:])
reframed = series_to_supervised(values,1,1)

countRow = len(reframed.values)
countFeature = len(reframed.columns)//2
curtime = datetime.datetime.strptime(reframed.iloc[-1,countFeature ], "%H:%M:%S")
nexttime = curtime + datetime.timedelta(minutes = 15)

if(datetime.time(0, 0) <= nexttime.time() and nexttime.time() <datetime.time(5,0)):
    new_period=nexttime.replace(hour=5, minute=00).strftime('%Y-%m-%d')

reframed.drop(reframed.columns[[0,countFeature]], axis=1, inplace=True)
#print(len(reframed.columns))
#print(countRow)
#scaled = scaler.fit_transform(reframed)

results = [countRow + 2,nexttime]




# drop columns we don't want to predict

#with open("frame1.csv",'wb') as rsFile:
with open("fram1.csv", "a", newline='') as myfile:

    wr = csv.writer(myfile,dialect='excel')
    wr.writerow(results)

# split into train and test sets
values = reframed.values
#n_train_fr = 30 * 24 * 95  #train = 1/4 test
n_train_fr = 4
train = values[:n_train_fr, :]
test = values[n_train_fr :, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print(train_X.shape[0],train_X.shape[1])
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
print(test_X,yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
mse = numpy.sqrt(mean_squared_error(inv_y, inv_yhat))
test_mse = tf.reduce_mean(tf.squared_difference(yhat, inv_y))
#rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(inv_y, inv_yhat))))
print(test_mse)
print('Test MSE: %.3f' % test_mse)
print('MSE: %.3f' % mse)
#print('Test RMSE: %.3f' % rmse)
print(results)
