#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Perform imports here:
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from datetime import datetime

# Import data
df=pd.read_csv('data/threeMonths.csv')

# Preprocessing
df['datee'] = pd.to_datetime(df['datee']).dt.date

print('Finding unique accounts')
unique_accounts=df['a_key'].append(df['b_key'],ignore_index=True).unique().tolist()
print('Done')
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
                         value=unique_accounts[7]
                         )
                    ]                        
                ),
                html.Div(
                    id='dummy',
                    style={'display': 'none'}
                ),
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Dropdown(
                         id='dropdown_2',
                         # options=[{'label': i, 'value': i} for i in unique_accounts],
                         # value=unique_accounts[3]
                         )
                    ]                        
                ),
                html.Div(
                    className="col-2",
                    children=[
                        dcc.DatePickerRange(
                            id='datepicker',
                            display_format='DD MM YYYY',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation='vertical',
                        )  
                    ]
                ),
                html.Div(
                    className="col-2",
                    children=[
                        html.Button(
                        id='submit-button',
                        children='Submit',
                        n_clicks=0
                        )
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
        )
])


# Define callbacks

@app.callback(
    [Output('dummy','children'),
     Output('dropdown_2','options'),
     Output('dropdown_2','value'),
     Output('datepicker','min_date_allowed'),
     Output('datepicker','max_date_allowed'),
     Output('datepicker','start_date'),
     Output('datepicker','end_date')],
    [Input('dropdown','value')])
def update_user_input_components(key):
    print('Updating user input components')
    
    df_filtered=filter_df(key)
    accounts_filtered=df_filtered['a_key'].append(df_filtered['b_key'],ignore_index=True)
    unique_accounts_filtered=accounts_filtered.unique()
    unique_accounts_filtered_without_key=np.delete(unique_accounts_filtered,np.where(unique_accounts_filtered==key))
    
    options=[{'label': i, 'value': i} for i in unique_accounts_filtered_without_key]
    
    value = unique_accounts_filtered_without_key[0]
    
    min_date_filtered=df_filtered['datee'].min()
    max_date_filtered=df_filtered['datee'].max()

    
    print('Done updating user input components')
    return key,options,value,min_date_filtered,max_date_filtered,min_date_filtered,max_date_filtered



@app.callback(
    [Output('timegraph','figure'),
     Output('freqgraph','figure')],
    [Input('submit-button', 'n_clicks')],
    [State('dummy','children'),
     State('dropdown_2','value'),
     State('datepicker','start_date'),
     State('datepicker','end_date')])
def update_timegraphs(n_clicks,key,key_2,start_date,end_date):
    print('Updating timegraphs')
    
    # if (not key or not key_2 or not start_date or not end_date): return
    if ((key is None) or (key_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing Update')
        raise PreventUpdate
    else:
        # Calculate necessary variables
        
        sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
        ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    
        df_key=filter_df(key,sd,ed).sort_values('datee')   
        df_key_2=filter_df(key_2,sd,ed).sort_values('datee')
        df_key_12=df_key[(df_key['a_key']==key_2) | (df_key['b_key']==key_2)].sort_values('datee')
            
        df_key['cum_amount']=df_key['boeking_eur'].cumsum()
        df_key_2['cum_amount']=df_key_2['boeking_eur'].cumsum()
        df_key_12['cum_amount']=df_key_2['boeking_eur'].cumsum()
        
        intervals=(df_key['datee']-df_key['datee'].shift(1)).dropna().apply(lambda x: x.days)
        intervals_2=(df_key_2['datee']-df_key_2['datee'].shift(1)).dropna().apply(lambda x: x.days)
        intervals_12=(df_key_12['datee']-df_key_12['datee'].shift(1)).dropna().apply(lambda x: x.days)
        
        # Update time graphs
        
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
                name='Transactions'
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
                name='Cumulative Transactions'
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
                name='Cumulative Transactions'
            ),
            row=2,
            col=1
        )
        
        # row 3
        if (not df_key_12.empty):
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
                    name='Cumulative Transactions'
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
            margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
            hovermode='closest'
        )
        
        # Update frequency graphs
      
        fig_freq = make_subplots(rows=3, cols=1, vertical_spacing=0.02)
         
        # row 1
        fig_freq.add_trace(go.Histogram(
                x=intervals,
                name='Intervals'
            ),
            row=1,
            col=1
        )
        
        # row 2
        fig_freq.add_trace(go.Histogram(
                x=intervals_2,
                name='Intervals'
            ),
            row=2,
            col=1
        )
        
        # row 3
        fig_freq.add_trace(go.Histogram(
                x=intervals_12,
                name='Intervals'
            ),
            row=3,
            col=1
        )
        
        # Update xaxis properties
        fig_freq.update_xaxes(title_text="Interval (days)", row=3, col=1)
    
        # Update yaxis properties
        fig_freq.update_yaxes(title_text="Frequency", row=1, col=1)
        fig_freq.update_yaxes(title_text="Frequency", row=2, col=1)
        fig_freq.update_yaxes(title_text="Frequency", row=3, col=1)
    
        # Update layout
        fig_freq.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
            hovermode='closest'
        )
        
        print('Done updating graphs')
        
        return fig,fig_freq

# Add the server clause:
if __name__ == '__main__':
    app.run_server()