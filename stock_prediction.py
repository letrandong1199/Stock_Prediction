import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import time

app = dash.Dash()
server = app.server
# Input data


def dataName(x):
    switcher = {
        'FB': 'FB',
        'TSLA': 'TSLA',
        'AAPL': 'AAPL',
        'MSFT': 'MSFT',
    }
    return switcher.get(x, "please choose name of data")


def dataType(x):
    switcher = {
        'Close': 'Close',
        'Price Rate Of Change': 'ROC',
    }
    return switcher.get(x, "please choose type of data")


def modelName(x):
    switcher = {
        'LSTM': 'lstm-model',
        'RNN': 'rnn-model',
        'XGBOOST': 'xgboost-model',
    }
    return switcher.get(x, "please choose model")


modelPath = {"lstm-model": "lstm_model.h5",
             "rnn-model": "rnn_model.h5", "xgboost-model": "xgboost_model.json"}


def InputData(df, value, modelPath, tye):
    
    if dataType(tye) == 'ROC':
        df['ROC'] = ((df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60)) * 100
        df['ROC'].fillna(0, inplace=True)
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']

    data = df.sort_index(ascending=True, axis=0)

    if dataType(tye) == 'ROC':
        new_dataset = pd.DataFrame(index = range(0, len(df)), columns = ['Date','Close', dataType(tye)])
        
    else:
        new_dataset = pd.DataFrame(index=range(
         0, len(df)), columns=['Date', dataType(tye)])
        


    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset[dataType(tye)][i] = data[dataType(tye)][i]
        
    print(new_dataset.head());

    if dataType(tye) == 'ROC':
        for i in range(0, len(data)):
            new_dataset["Close"][i] = data["Close"][i]        # Combine two columns with addition
        new_dataset['ROC'] = new_dataset['Close'] + new_dataset['ROC']
        plot_dataset = new_dataset.copy() # For plotting

        new_dataset.drop('Close', axis=1, inplace=True)

        
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Erase Date column
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    # Get Close column values
    final_dataset = new_dataset.values

    train_data = final_dataset[0: 987, :]  # To train
    valid_data = final_dataset[987:, :]  # To test

    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale
    scaled_data = scaler.fit_transform(final_dataset)  # Activate the scale

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60: i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    inputs = new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #print(value)
    if value == 'XGBOOST':
        start = time.time()
        x_train_data = np.squeeze(x_train_data, 2)
        model = XGBRegressor(objective='reg:squarederror',
                             verbose=False)
        params = {'gamma': 0.001, 'learning_rate': 0.05, 'max_depth': 15, 'n_estimators': 300, 'random_state': 103}
        model = XGBRegressor(**params, objective='reg:squarederror')
        model.fit(x_train_data, y_train_data, verbose=True)
       
        end = time.time()
        print('Took: {}s'.format(end - start))
        X_test = np.squeeze(X_test, 2)
    else:
        model = load_model(modelPath[modelName(value)])

    closing_price = model.predict(X_test)
    if value == 'XGBOOST':
        closing_price = np.expand_dims(closing_price, 1)
    closing_price = scaler.inverse_transform(closing_price)

    train_data = new_dataset[:987]
    valid_data = new_dataset[987:]
    valid_data['Predictions'] = closing_price

    return train_data, valid_data, new_dataset


df_fb = pd.read_csv("./stock_data.csv")

df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")

# Data stock_data

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'LSTM-model', 'value': 'LSTM'},
            {'label': 'RNN-model', 'value': 'RNN'},
            {'label': 'XGBOOST-model', 'value': 'XGBOOST'},
        ],
        value='LSTM'
    ),
    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Stock NES Data",
                        style={"textAlign": "center"}),
                html.H3("Type of data",
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='type-dropdown',
                             options=[{'label': 'Price Rate Of Change', 'value': 'Price Rate Of Change'},
                                      {'label': 'Close', 'value': 'Close'}],
                             multi=True, value=['Close'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(
                    id="Actual Data",
                ),
                
            ])


        ]),
        dcc.Tab(label='Stock Facebook Data', children=[
            html.Div([
                html.H1("Facebook Stocks Closing Price",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                html.H3("Type of data",
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='type-dropdown-fb',
                             options=[{'label': 'Price Rate Of Change', 'value': 'Price Rate Of Change'},
                                      {'label': 'Close', 'value': 'Close'}],
                             multi=True, value=['Close'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                
            ], className="container"),
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value'),#FB/Testla/apple
              Input('model-dropdown', 'value'),#LSTM/RNN/XGBOOST
              Input('type-dropdown-fb', 'value')])#Close/ROC
def update_graph(selected_dropdown, selected_dropdown_model, select_dropdown_type):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
                "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []
    trace3 = []
    for stock in selected_dropdown:
        #print('stock', stock)
        df = df_fb[df_fb['Stock'] == dataName(stock)]
        #print(df)
        for tye in select_dropdown_type:
            result = InputData(df, selected_dropdown_model, modelPath, tye)
            #print('result', result)
            train = result[0]
            valid = result[1]
            new_data = result[2]
            trace1.append(
                go.Scatter(x=new_data.index,
                           y=new_data[dataType(tye)],
                           mode='lines',
                           name=f'Data train {tye} {dropdown[stock]}', textposition='bottom center'))
            trace2.append(
                go.Scatter(x=valid.index,
                           y=valid[dataType(tye)],
                           mode='lines',
                           name=f'Actual {tye} {dropdown[stock]}', textposition='bottom center'))
            trace3.append(
                go.Scatter(x=valid.index,
                           y=valid['Predictions'],
                           mode='lines',
                           name=f'Predicted {tye} {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"{tye} Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure
############################################################


@app.callback(Output('Actual Data', 'figure'),
              [dash.dependencies.Input('model-dropdown', 'value'),
              dash.dependencies.Input('type-dropdown', 'value')])
def update_graph(selected_dropdown_model, select_dropdown_type):
    trace1 = []
    trace2 = []
    trace3 = []
    for tye in select_dropdown_type:
        result = InputData(df_nse, selected_dropdown_model, modelPath, tye)
        train = result[0]
        valid = result[1]
        new_data = result[2]
        trace1.append(
            go.Scatter(x=new_data.index,
                       y=new_data[dataType(tye)],
                       mode='lines',
                       name=f'Data train {tye}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=valid.index,
                       y=valid[dataType(tye)],
                       mode='lines',
                       name=f'Actual {tye}', textposition='bottom center'))
        trace3.append(
            go.Scatter(x=valid.index,
                       y=valid['Predictions'],
                       mode='lines',
                       name=f'Predicted {tye}', textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"{select_dropdown_type} for NSE data using {selected_dropdown_model}-model",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])}},
                                  yaxis={"title": "Price (USD)"})}
    return figure




if __name__ == '__main__':
    app.run_server(debug=True)
