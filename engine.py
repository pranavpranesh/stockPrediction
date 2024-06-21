
import streamlit as st

import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

from datetime import datetime

from keras.models import Sequential

from keras.layers import Dense, SimpleRNN
 
# Load and preprocess Abalone data

@st.cache_data

def load_abalone_data():

    abalone_features = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight",

                        "Viscera weight", "Shell weight", "Age"]

    abalone_train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",

                                names=abalone_features)

    abalone_test = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_test.csv",

                               names=abalone_features)

    return abalone_train, abalone_test
 
abalone_train, abalone_test = load_abalone_data()

abalone_train_features = abalone_train.copy()

abalone_train_labels = abalone_train_features.pop('Age')

abalone_test_features = abalone_test.copy()

abalone_test_labels = abalone_test_features.pop('Age')
 
normalize = tf.keras.layers.Normalization()

normalize.adapt(abalone_train_features)
 
# Abalone model definition

def build_abalone_model():

    model = tf.keras.Sequential([

        normalize,

        tf.keras.layers.Dense(64),

        tf.keras.layers.Dense(1)

    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),

                  optimizer=tf.keras.optimizers.Adam())

    return model
 
# Streamlit app

st.title('Model Prediction App')
 
model_choice = st.selectbox('Choose a model', ['Abalone Model', 'Stock RNN Model'])
 
if model_choice == 'Abalone Model':

    model = build_abalone_model()

    model.fit(abalone_train_features, abalone_train_labels, epochs=10, verbose=0)

    predictions = model.predict(abalone_test_features)

    mse = mean_squared_error(abalone_test_labels, predictions)

    st.write('Mean Squared Error:', mse)

    st.write('Predictions:', predictions[:10])  # Display first 10 predictions
 
elif model_choice == 'Stock RNN Model':

    # User input for stock ticker and date range

    st.write("Select stock and date range for prediction")

    ticker = st.text_input('Stock Ticker', 'AAPL')

    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))

    end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))
 
    if st.button('Load Data'):

        # Load and preprocess stock data

        @st.cache_data

        def load_stock_data(ticker, start_date, end_date):

            dataset = yf.download(ticker, start=start_date, end=end_date)

            tstart = '2016-01-01'

            tend = '2020-12-31'

            train = dataset.loc[tstart:tend, ['High']].values

            test = dataset.loc['2021-01-01':, ['High']].values

            return train, test
 
        train, test = load_stock_data(ticker, start_date, end_date)

        sc = MinMaxScaler(feature_range=(0, 1))

        train = train.reshape(-1, 1)

        train_scaled = sc.fit_transform(train)

        n_steps = 1

        features = 1
 
        def split_sequence(sequence, n_steps):

            X, y = list(), list()

            for i in range(len(sequence)):

                end_ix = i + n_steps

                if end_ix > len(sequence) - 1:

                    break

                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

                X.append(seq_x)

                y.append(seq_y)

            return np.array(X), np.array(y)
 
        X_train, y_train = split_sequence(train_scaled, n_steps)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)
 
        def prepare_test_data(test, n_steps):

            test_scaled = sc.transform(test.reshape(-1, 1))

            X_test, y_test = split_sequence(test_scaled, n_steps)

            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

            return X_test, y_test
 
        X_test, y_test = prepare_test_data(test, n_steps)
 
        # Stock RNN model definition

        def build_stock_rnn_model():

            model = Sequential()

            model.add(SimpleRNN(units=125, input_shape=(n_steps, features)))

            model.add(Dense(units=1))

            model.compile(optimizer="RMSprop", loss="mse")

            return model
 
        model = build_stock_rnn_model()

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predictions = model.predict(X_test)

        predictions = sc.inverse_transform(predictions)

        mse = mean_squared_error(test[:len(predictions)], predictions)

        st.write('Mean Squared Error:', mse)

        st.write('Predictions:', predictions[:10])  # Display first 10 predictions
 
        # Plot predictions

        fig, ax = plt.subplots()

        ax.plot(test[:len(predictions)], color="gray", label="Real")

        ax.plot(predictions, color="red", label="Predicted")

        ax.set_title('Stock Price Prediction')

        ax.set_xlabel("Time")

        ax.set_ylabel("Stock Price")

        ax.legend()

        st.pyplot(fig)
