#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Perform imports here:
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from datetime import datetime

# Import data
df=pd.read_csv('data/threeMonths.csv')

# Preprocessing
df['datee'] = pd.to_datetime(df['datee']).dt.date

unique_accounts=df['a_key'].unique().tolist()
unique_accounts.sort()
min_date=df['datee'].min()
max_date=df['datee'].max()

def filter_df(acc_key,start_date=min_date,end_date=max_date):
    condition=(df['datee']>=start_date) & (df['datee']<=end_date) & ((df['a_key']==acc_key) | (df['b_key']==acc_key))
    return df[condition]

# Launch the application:
app = dash.Dash(__name__)


# Set the app layout
app.layout=html.Div([
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Dropdown(
                         id='dropdown',
                         options=[{'label': i, 'value': i} for i in unique_accounts],
                         value=unique_accounts[0]
                         )
                    ]                        
                ),
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Dropdown(
                         id='dropdown_2',
                         options=[{'label': i, 'value': i} for i in unique_accounts],
                         value=unique_accounts[0]
                         )
                    ]                        
                ),
                html.Div(
                    className="col-2",
                    id='datepicker_div',
                    children=[
                        dcc.DatePickerRange(
                            id='datepicker',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
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
                                )
                            ]),
                            dcc.Tab(label='Frequency', children=[
                                dcc.Graph(
                                    id='freqgraph'
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
    [Output('dropdown_2','options'),Output('datepicker_div', 'children')],
    [Input('dropdown','value')])
def update_user_input_components(account_key):
    df_filtered=filter_df(account_key)
    accounts_filtered=df_filtered['a_key'].append(df_filtered['b_key'],ignore_index=True)
    options=[{'label': i, 'value': i} for i in accounts_filtered.unique()]
    
    min_date_filtered=df_filtered['datee'].min()
    max_date_filtered=df_filtered['datee'].max()
    datepicker=dcc.DatePickerRange(
        id='datepicker',
        min_date_allowed=min_date_filtered,
        max_date_allowed=max_date_filtered,
        start_date=min_date_filtered,
        end_date=max_date_filtered,
        display_format='DD MM YYYY',
        start_date_placeholder_text="Start Date",
        end_date_placeholder_text="End Date",
        calendar_orientation='vertical',
    )
    
    return options,datepicker



@app.callback(
    Output('timegraph','figure'),
    [Input('dropdown','value'),
     Input('dropdown_2','value'),
     Input('datepicker','start_date'),
     Input('datepicker','end_date')])
def update_timegraphs(key,key_2,start_date,end_date):
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()

    df_key=filter_df(key,sd,ed).sort_values('datee')   
    df_key_2=filter_df(key_2,sd,ed).sort_values('datee')
    df_key_12=df_key[(df_key['a_key']==key_2) | (df_key['b_key']==key_2)].sort_values('datee')
        
    df_key['cum_amount']=df_key['boeking_eur'].cumsum()
    df_key_2['cum_amount']=df_key_2['boeking_eur'].cumsum()
    df_key_12['cum_amount']=df_key_2['boeking_eur'].cumsum()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    # row 1
    fig.add_trace(
        go.Scatter(
            x=df_key['datee'],
            y=df_key['boeking_eur'],
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Cumulative Amount'
        ),     
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_key['datee'],
            y=df_key['cum_amount'],
            mode='markers+lines',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),     
        row=1,
        col=1
    )

    # row 2
    fig.add_trace(go.Scatter(
            x=df_key_2['datee'],
            y=df_key_2['boeking_eur'],
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=2,
        col=1
    )
    fig.add_trace(go.Scatter(
            x=df_key_2['datee'],
            y=df_key_2['cum_amount'],
            mode='markers+lines',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=2,
        col=1
    )
    
    # row 3
    fig.add_trace(go.Scatter(
            x=df_key_12['datee'],
            y=df_key_12['boeking_eur'],
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=3,
        col=1
    )
    fig.add_trace(go.Scatter(
            x=df_key_12['datee'],
            y=df_key_12['cum_amount'],
            mode='markers+lines',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=3,
        col=1
    )
    
    # Update xaxis properties
    fig.update_xaxes(title_text="Time", row=3, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    fig.update_yaxes(title_text="Amount", row=3, col=1)

    # Update layout
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
    )

    return fig


@app.callback(
    Output('freqgraph','figure'),
    [Input('dropdown','value'),
     Input('dropdown_2','value'),
     Input('datepicker','start_date'),
     Input('datepicker','end_date')])
def update_freqgraphs(key,key_2,start_date,end_date):
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()

    df_key=filter_df(key,sd,ed).sort_values('datee')   
    df_key_2=filter_df(key_2,sd,ed).sort_values('datee')
    df_key_12=df_key[(df_key['a_key']==key_2) | (df_key['b_key']==key_2)].sort_values('datee')
    
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.02)
    
      
    intervals=df_key['datee']-df_key['datee'].shift(1)
    intervals_2=df_key_2['datee']-df_key_2['datee'].shift(1)
    intervals_12=df_key_12['datee']-df_key_12['datee'].shift(1)

    # row 1
    fig.add_trace(go.Histogram(
            x=intervals,
            name='Intervals'
        ),
        row=1,
        col=1
    )
    
    # row 2
    fig.add_trace(go.Histogram(
            x=intervals_2,
            name='Intervals'
        ),
        row=2,
        col=1
    )
    
    # row 3
    fig.add_trace(go.Histogram(
            x=intervals_12,
            name='Intervals'
        ),
        row=3,
        col=1
    )
    
    # Update xaxis properties
    fig.update_xaxes(title_text="Bins (days)", row=3, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)

    # Update layout
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
        hovermode='closest'
    )

    return fig

# Add the server clause:
if __name__ == '__main__':
    app.run_server()