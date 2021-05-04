#Daniel Holland
#imports libarys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as plt
import matplotlib.pyplot as plt 
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout


#gives the user information on the SPY index
print("Entering the company SPY, will give you a prediction on the S&P 500  ")


#asks the user for a company ticker symbol
company = input("enter company ticker symbol: ")
#sets start date
start = dt.datetime(2014,1,1)
#sets star date
end = dt.datetime(2020,1,1,)

#creates a loop to ensure data is collected
for x in range(1):
        try:
            df = web.DataReader(company, 'yahoo', start, end)
            df.shape
            print("data collected")
            break
        #checks for input error
        except KeyError:
            print("no data from this company was found in 2014")
            False


#shows information to the user
print("The technical indicator being used, is 'close'.\nThe closing price is the last price at which the stock traded during the regular trading day")



#shows information to the user
print("The algorithm being used is a recurrent neural network, that uses long short-term memory(LSTM)\nThe algorithm uses the last 60 days of data to predict the next day.\nThis predicted day is then used to make the next predicted day and so on.  ")



#sets the range to scale the data
scaler = MinMaxScaler(feature_range = (0,1))

#scales the close data
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

scaled_data.shape


#sets last_days to 60
last_days = 60







#creates 2 lists
x_train = []
y_train = []



#creates a loop to retrive the 60 days
for x in range(last_days, len(scaled_data)):
    #adds them to the list
    x_train.append(scaled_data[x-last_days:x, 0])
    y_train.append(scaled_data[x, 0])





#sets the list to arrays
x_train, y_train = np.array(x_train), np.array(y_train)






#reshapes x_train
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))




#sets model to sequential
model = Sequential()
#creates first layer and variables set
model.add(LSTM(units = 55,return_sequences = True, input_shape = (x_train.shape[1], 1)))
#dropsout set to 0.2
model.add(Dropout(0.2))
#creates first layer and variables set
model.add(LSTM(units = 55,return_sequences = True))
#dropsout set to 0.3
model.add(Dropout(0.30))

#creates first layer and variables set
model.add(LSTM(units = 55))
#dropsout set to 0.3
model.add(Dropout(0.50))
#
model.add(Dense(units = 1))



#model compiled
model.compile(optimizer = 'adam', loss = 'mean_squared_error')





#model was then trained with different paramaters
model.fit(x_train, y_train, epochs = 5, batch_size = 10)



# test data
#sets dates
start_test = dt.datetime(2020,1,1)
end_test = dt.datetime.now()
#gets data
test_data = web.DataReader(company, 'yahoo', start_test, end_test)

test_data



#gets only the close data
actual_prices = test_data['Close'].values




#concats the data
total_dataset = pd.concat((df['Close'], test_data['Close']), axis = 0)



#further pre-processes test data
model_inputs = total_dataset[len(total_dataset) - len(test_data) - last_days:].values


#reshapes and scales the data
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)



#creates empty list
x_test = []



#sets the loop
for x in range(last_days, len(model_inputs)):
    x_test.append(model_inputs[x-last_days:x, 0])



x_test


#sets as array
x_test = np.array(x_test)


reshapes
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
x_test.shape


#makes the prediction
predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)


#visulizes the data
plt.plot(actual_prices, color='red', label=f"actual {company} Price")
plt.plot(predicted, color='blue',label=f"Predicted {company} Price")
plt.title(f" {company} Share Price")
plt.xlabel('Time')
plt.ylabel(f' {company} Share Price')
plt.legend()
plt.show()


#sets the dataset for the next day prediction
future = [model_inputs[len(model_inputs) + 1 - last_days:len(model_inputs+1), 0]]
future = np.array(future)
future = np.reshape(future,(future.shape[0], future.shape[1],1))



#produces the prediction
prediction = model.predict(future)
prediction = scaler.inverse_transform(prediction)
#prints prediction
print(f"The Prediction for tomorrow for {company} :{prediction}")
























