# Import necessary libraries
import yfinance as yf
import numpy as np
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from binance import fetch_binance_data

from datetime import datetime

scaler = MinMaxScaler()


def get_data():
    try:
        data = pd.read_csv("data.csv")
        return data
    except:  # noqa: E722
        data = fetch_binance_data()
        data["ROC"] = data["Close"].pct_change(periods=1)
        data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
        data["Moving_Average"] = data["Close"].rolling(window=14).mean()
        data = data.dropna()
        # save to csv file
        data.to_csv("data.csv")
        return data


def visualize_data(data, column):
    plt.figure(figsize=(14, 7))
    plt.title(f"{column} Stock Price History")
    plt.plot(data[column], label=column)
    plt.xlabel("Year")
    plt.ylabel(f"{column} Price USD ($)")
    plt.show()


def create_sequences(data, column_index, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back : i, :])
        y.append(
            data[i, column_index]
        )  # predicting the target_column (Close price, ROC, etc.)
    return np.array(x), np.array(y)


def build_and_train_model(x_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(x_train.shape[1], x_train.shape[2]),
        )
    )
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def preprocess_data(data, target_column):
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def run_main(target_column):
    data = get_data()

    scaled_data = preprocess_data(data, target_column)
    column_index = data.columns.get_loc(target_column)
    x, y = create_sequences(scaled_data, column_index)

    x_train, x_test = x[: int(len(x) * 0.8)], x[int(len(x) * 0.8) :]
    y_train, y_test = y[: int(len(y) * 0.8)], y[int(len(y) * 0.8) :]

    try:
        model = load_model(f"predict_{target_column}_model.h5")
    except:  # noqa: E722
        model = build_and_train_model(x_train, y_train)
        model.save(f"predict_{target_column}_model.h5")
    predictions = model.predict(x_test)

    # inverse_predict = scaler.inverse_transform(predictions[["Close"]])
    # inverse_y_test = scaler.inverse_transform(y_test[["Close"]])
    # create a dataframe to store the predictions and actual values
    df = pd.DataFrame()
    # format date to DD-MM-YYYY
    df["Date"] = data.index[-len(predictions) :]
    df["Predictions"] = predictions
    df["Actual"] = y_test
    # df["Inverse_Predictions"] = scaler.inverse_transform(predictions)
    # df["Inverse_Actual"] = scaler.inverse_transform(y_test)
    print(df)
    return df


app = dash.Dash("Stock Predictor")
server = app.server

app.layout = html.Div(
    [
        html.H1("Stock Predictor"),
        dcc.Dropdown(
            id="chart-dropdown",
            options=[
                {"label": "Close", "value": "close"},
                {"label": "ROC", "value": "roc"},
                {"label": "MA", "value": "ma"},
                {"label": "RSI", "value": "rsi"},
            ],
            value="close",  # Default value
        ),
        dcc.Graph(id="chart-output"),
    ]
)


@app.callback(Output("chart-output", "figure"), [Input("chart-dropdown", "value")])
def update_chart(selected_chart):
    fig = go.Figure()
    if selected_chart == "roc":
        df = run_main("ROC")
        return draw_chart(df)

    elif selected_chart == "close":
        df = run_main("Close")
        return draw_chart(df)
    elif selected_chart == "ma":
        df = run_main("Moving_Average")
        return draw_chart(df)
    elif selected_chart == "rsi":
        df = run_main("RSI")
        return draw_chart(df)
    return fig


def draw_chart(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Predictions"], mode="lines", name="Predictions")
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], mode="lines", name="Actual"))
    return fig


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
