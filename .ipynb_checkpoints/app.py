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
API_KEY = 'YOUR_API_KEY'

# Create a TimeSeries object with your API key
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Define the symbol and data interval
symbol = 'AAPL'  # Example symbol (Apple Inc.)
interval = 'daily'  # Data interval: 'daily', 'weekly', 'monthly'
st.title('Stock Trend Application')
user_input=st.text_input('Enter Stock Symbol', 'AAPL')
# Retrieve stock price data
df, meta_data = ts.get_daily(user_input, outputsize='full')


