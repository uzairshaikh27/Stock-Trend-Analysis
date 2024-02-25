import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
import sklearn
import tensorflow as tf
from keras.models import load_model
import streamlit as st


from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = 'LTR51FIS2KUJIGN7'

# Create a TimeSeries object with your API key
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Define the symbol and data interval
symbol = 'AAPL'  # Example symbol (Apple Inc.)
interval = 'daily'  # Data interval: 'daily', 'weekly', 'monthly'
def set_custom_theme():
    st.markdown(
        """
        <style>
        .css-1k3d8rh {
            color: #ccc;
        }
        .css-1g6xa3u {
            color: #fff;
        }
        .st-d7 {
            background-color: #1e1e1e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_theme()
st.title('Stock Trend Application')

user_input=st.text_input('Enter Stock Symbol', 'AAPL')
# Retrieve stock price data
df, meta_data = ts.get_daily(user_input, outputsize='full')
st.write(df.describe())

st.subheader('Closing Price vs Time')

fig=plt.figure(figsize=(12,6))
plt.plot(df["4. close"])
st.pyplot(fig)

# ma100=df["4. close"].rolling(100).mean()
# fig=plt.figure(figsize=(12,6))
# # plt.plot(ma100)
# plt.plot(df["4. close"])
# st.pyplot(fig)


st.subheader('Closing Price with MA100 and MA200  vs Time')
ma100=df["4. close"].rolling(100).mean()

ma200=df['4. close'].rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(ma200,'g',label='MA200')
plt.legend()
plt.plot(df["4. close"],'b',label='Original Price')
st.pyplot(fig)



data_training=pd.DataFrame(df['4. close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['4. close'][int(len(df)*0.70):int(len(df))])\

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training )
data_training_array

# x_train=[]
# y_train=[]
# for i in range(100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
# x_train,y_train=np.array(x_train),np.array(y_train)

# model.save('my_model.h5')
model=load_model('my_model.h5')
model.save('my_model.h5')
#t
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]): 
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)


scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Prediction vs Original Price')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()  
st.pyplot(fig2)