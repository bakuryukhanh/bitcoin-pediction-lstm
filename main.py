# Import necessary libraries
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from binance import fetch_binance_data

import dash_bootstrap_components as dbc


scaler = MinMaxScaler()
n_lookback = 100
n_forecast = 30
epochs = 100


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


def build_and_train_model(x_train, y_train, epochs=50, batch_size=32):
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
        )
    )
    model.add(LSTM(units=50))

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def preprocess_data(data):
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def format_train_test(data, target_column):
    x = []
    y = []

    if target_column == "ROC" or target_column == "RSI":
        for i in range(0, data.shape[0]):
            x.append(data[i].reshape(1, -1))
            y.append(data[i, :])
    else:
        for i in range(n_lookback, len(data)):
            x.append(data[i - n_lookback : i])
            y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    return x, y


def predict(target_column, retrain=False):
    origin_data = get_data()
    data = origin_data[[target_column]]
    data = data.dropna()

    data_train = data[: int(len(data) * 0.8)]
    data_test = data[int(len(data) * 0.8) :]

    scaled_data = preprocess_data(data_train)
    x_train, y_train = format_train_test(scaled_data, target_column)

    model = None
    try:
        if retrain:
            raise Exception("Retrain model")
        model = load_model(f"predict_{target_column}_model.h5")
    except:  # noqa: E722
        model = build_and_train_model(x_train, y_train, epochs=epochs)
        model.save(f"predict_{target_column}_model.h5")

    # test the model
    scaled_data = preprocess_data(data_test)
    x_test, y_test = format_train_test(scaled_data, target_column)
    predictions = model.predict(x_test)

    # create future prediction x
    scaled_data = preprocess_data(data_train)
    # x_future = scaled_data[-n_lookback:]
    # x_future = x_future.reshape(1, n_lookback, 1)
    # print(x_future.shape)
    # future_predictions = model.predict(x_future).reshape(-1)
    # future_predictions = scaler.inverse_transform(
    #     future_predictions.reshape(-1, 1)
    # ).reshape(-1)
    # print(future_predictions)

    df = pd.DataFrame()

    df["Date"] = origin_data["date_string"]
    df["Actual"] = origin_data[target_column]
    prediction_index = len(origin_data) - len(predictions)
    df["Predictions"] = np.nan
    df["Predictions"].iloc[prediction_index : len(origin_data)] = (
        scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    )

    df_future = pd.DataFrame()
    df_future["Date"] = pd.date_range(
        start=origin_data["date_string"].iloc[-1], periods=n_forecast, freq="d"
    )
    print(df_future)
    # df_future["Future Predictions"] = future_predictions

    result = pd.concat([df, df_future])

    # append date for future predictions
    # df["Date"] = pd.concat([df["Date"], future_dates], ignore_index=True)
    return result


app = dash.Dash("Stock Predictor", external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
df = get_data()
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["date_string"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    ]
)


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Stock Predictor"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Select chart to display:"),
                                dcc.Dropdown(
                                    id="chart-dropdown",
                                    options=[
                                        {"label": "Close", "value": "close"},
                                        {"label": "ROC", "value": "roc"},
                                        {"label": "MA", "value": "ma"},
                                        {"label": "RSI", "value": "rsi"},
                                    ],
                                    value="close",  # Default value
                                    style={"width": "250px"},
                                ),
                            ],
                            className="d-flex gap-2 align-items-center",
                        ),
                        html.Button(
                            "Update Data & Retrain model",
                            id="update-data",
                            n_clicks=0,
                            className="btn btn-primary my-2",
                        ),
                    ],
                    className="d-flex justify-content-between align-items-center",
                ),
                dcc.Graph(id="data-chart", figure=fig),
                dcc.Loading(
                    [dcc.Graph(id="chart-output")],
                    overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                ),
            ],
            className="p-5",
        ),
    ]
)


@app.callback(Output("chart-output", "figure"), [Input("chart-dropdown", "value")])
def update_chart(selected_chart):
    fig = go.Figure()
    if selected_chart == "roc":
        df = predict("ROC")
        return draw_chart(df, "ROC")

    elif selected_chart == "close":
        df = predict("Close")
        return draw_chart(df, "Close")
    elif selected_chart == "ma":
        df = predict("Moving_Average")
        return draw_chart(df, "Moving Average")
    elif selected_chart == "rsi":
        df = predict("RSI")
        return draw_chart(df, "RSI")
    return fig


def draw_chart(df, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Predictions"], mode="lines", name="Predictions")
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], mode="lines", name="Actual"))
    # fig.add_trace(
    #     go.Scatter(
    #         x=df["Date"],
    #         y=df["Future Predictions"],
    #         mode="lines",
    #         name="Future Predictions",
    #     )
    # )
    fig.update_layout(title=title)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=title)
    return fig


# @app.callback(Output("chart-output", "test"), [Input("update-data", "n_clicks")])
# def update_data(n_clicks):
#     if n_clicks > 0:
#         data = fetch_binance_data()
#         data["ROC"] = data["Close"].pct_change(periods=1)
#         data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
#         data["Moving_Average"] = data["Close"].rolling(window=14).mean()
#         data = data.dropna()
#         # save to csv file
#         data.to_csv("data.csv")

#         predict("Close", retrain=True)
#         predict("ROC", retrain=True)
#         predict("RSI", retrain=True)
#         predict("Moving_Average", retrain=True)


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
