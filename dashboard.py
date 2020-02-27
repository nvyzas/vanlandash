#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Perform imports here:
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import numpy as np
import pandas as pd

df=pd.read_csv('data/data_preprocessed.csv')
df['date_tech'] = pd.to_datetime(df['date_tech']).dt.date

unique_accounts=df['a_key'].unique().tolist()
unique_accounts.sort()

# Launch the application:
app = dash.Dash()

# Set the app layout
app.layout=html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in unique_accounts],
        value=unique_accounts[0]
    ),
    html.H1(
        id='output-text',
    ),
    dcc.Graph(
        id='timegraph'
    ),
    dcc.Graph(
        id='freqgraph'
    )
 ])


# Define callbacks

@app.callback(
    Output('timegraph','figure'),
    [Input('dropdown','value')])
def update_timegraph(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('date_tech')
    df_key['cum_amount']=df_key['boeking_eur'].cumsum()
    
    data=[
        go.Scatter(
            x=df_key['date_tech'],
            y=df_key['boeking_eur'],
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.3,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        go.Scatter(
            x = df_key['date_tech'],
            y = df_key['cum_amount'],
            mode = 'markers+lines',
            marker={
            # 'size': 2.5,
            # 'opacity': 1,
            # 'line': {'width': 0.5, 'color': 'white'}
            },
            name = 'Cumulative amount'
        )
    ]
    
    layout= go.Layout(
        xaxis={'title': 'Date'},
        yaxis={'title': 'Amount'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )

    return {'data':data,'layout':layout}

@app.callback(
    Output('freqgraph','figure'),
    [Input('dropdown','value')])
def update_freqgraph(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('date_tech')
    intervals=df_key['date_tech']-df_key['date_tech'].shift(1)
    
    data=[
        go.Histogram(
            x=intervals.dt.days
            )
        ]
    
    layout= go.Layout(
        xaxis={'title': 'Interval (days)'},
        yaxis={'title': 'Frequency'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    
    return {'data':data,'layout':layout}

# Add the server clause:
if __name__ == '__main__':
    app.run_server()