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
from datetime import datetime as dt

# Import data
df=pd.read_csv('data/threeMonths.csv')

# Preprocessing
df['datee'] = pd.to_datetime(df['datee']).dt.date

unique_accounts=df['a_key'].unique().tolist()
unique_accounts.sort()


# Launch the application:
app = dash.Dash(__name__)


# Set the app layout
app.layout=html.Div([
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col-3",
                    children=[
                        dcc.Dropdown(
                         id='dropdown',
                         options=[{'label': i, 'value': i} for i in unique_accounts],
                         value=unique_accounts[0]
                         )
                    ]                        
                ),
                html.Div(
                    className="col-3",
                    children=[
                        dcc.Dropdown(
                         id='dropdown_2',
                         options=[{'label': i, 'value': i} for i in unique_accounts],
                         value=unique_accounts[0]
                         )
                    ]                        
                ),
                html.Div(
                    className="col-6",
                    children=[
                        dcc.DatePickerRange(
                            min_date_allowed=df['datee'].min(),
                            max_date_allowed=df['datee'].max(),
                            start_date=df['datee'].min(),
                            end_date=df['datee'].max(),
                            display_format='DD MM YYYY',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation='vertical',
                        )  
                    ]
                ),
                html.Div(
                    className='col-6',
                    children=[
                        dcc.Tabs([
                            dcc.Tab(label='Time', children=[
                                dcc.Graph(
                                    id='timegraph'
                                ),
                                dcc.Graph(
                                    id='timegraph_2'
                                ),
                                dcc.Graph(
                                    id='timegraph_3',
                                    figure={
                                        'data': [
                                            {'x': [1, 2, 3], 'y': [4, 1, 2],
                                                'type': 'bar', 'name': 'SF'},
                                            {'x': [1, 2, 3], 'y': [2, 4, 5],
                                             'type': 'bar', 'name': u'Montréal'},
                                        ]
                                    }
                                )
                            ]),
                            dcc.Tab(label='Frequency', children=[
                                dcc.Graph(
                                    id='freqgraph',
                                    figure={
                                        'data': [
                                            {'x': [1, 2, 3], 'y': [1, 4, 1],
                                                'type': 'bar', 'name': 'SF'},
                                            {'x': [1, 2, 3], 'y': [1, 2, 3],
                                             'type': 'bar', 'name': u'Montréal'},
                                        ]
                                    }
                                )
                            ])
                        ])
                    ]
                )
            ]
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className='col-6',
                    children='Network Graph Here'
                )
            ]
        )
])


# Define callbacks
@app.callback(
    Output('dropdown_2','options'),
    [Input('dropdown','value')])
def update_dropdown(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('datee')
    accounts_key=df_key['a_key'].append(df_key['b_key'],ignore_index=True)
    options=[{'label': i, 'value': i} for i in accounts_key.unique()]
    return options



@app.callback(
    Output('timegraph','figure'),
    [Input('dropdown','value')])
def update_timegraph(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('datee')
    df_key['cum_amount']=df_key['boeking_eur'].cumsum()
    
    data=[
        go.Scatter(
            x=df_key['datee'],
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
            x = df_key['datee'],
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
    Output('timegraph_2','figure'),
    [Input('dropdown_2','value')])
def update_timegraph_2(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('datee')
    df_key['cum_amount']=df_key['boeking_eur'].cumsum()
    
    data=[
        go.Scatter(
            x=df_key['datee'],
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
            x = df_key['datee'],
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

def update_timegraph_3(key):
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('datee')
    df_key['cum_amount']=df_key['boeking_eur'].cumsum()
    
    data=[
        go.Scatter(
            x=df_key['datee'],
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
            x = df_key['datee'],
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
    df_key=df[(df['a_key']==key) | (df['b_key']==key)].sort_values('datee')
    intervals=df_key['datee']-df_key['datee'].shift(1)
    
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