#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Perform imports here:
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
from colour import Color
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import time

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pyvis
#from pyvis.network import Network
from pyvis import network as net

df=pd.read_csv('data_shiva.csv')
#data1=pd.read_csv('data/shiva.csv')
node1 = pd.read_csv('nodes.csv')

df['date_tech'] = pd.to_datetime(df['datee']).dt.date
#edge1.dtypes

unique_accounts=df['originn'].unique().tolist()
unique_accounts.sort()

# Launch the application:
app = dash.Dash()

# Define callbacks nikos
MONTH=[20,25]
ACCOUNT="B642168"
################### Adding the nwtwork graph ########################
    


###########################LAYOUT#####################################

app.layout=html.Div([
    html.Div(
        className="row",
        children=[
            html.Div(
                className="six columns",
                children=[
                    html.Div(
                        children=dcc.Dropdown(
                             id='dropdown',
                             options=[{'label': i, 'value': i} for i in unique_accounts],
                             value=unique_accounts[0]
                        )
                    ),
                    html.Div(
                        children=dcc.RangeSlider(
                            id='my-range-slider',
                            min=20,
                            max=25,
                            step=1,
                            value=[22, 22], # default slider values
                            marks={
                                20: {'label': '20'},
                                21: {'label': '21'},
                                22: {'label': '22'},
                                23: {'label': '23'},
                                24: {'label': '24'},
                                25: {'label': '25'}
                               
                            }
                        )
					)
                ]
            ),
            html.Div(
                className="six columns",
                children=html.Div([
                    #dcc.Input(id='my-range-slider', value=[22,23]),
                    #dcc.Input(id='dropdown', value="B642168"),
                    #dcc.Graph(id="my-graph",figure=network_graph(value, value)),
                    html.Iframe(id='mapx', width='100%', height='550')
                ])
            )
        ]
    )
])


#####################################################################
'''
@app.callback(
      dash.dependencies.Output('mapx', 'srcDoc'),
    [dash.dependencies.Input('my-range-slider', 'value1')])

def update_output(value):
        return network_graph(value,ACCOUNT)
'''    
@app.callback(Output('mapx','srcDoc'), [Input('my-range-slider','value'), Input('dropdown', 'value')])
def update_output_div(monthRange,Account):
    bank_data = df.copy()
    
    granular_bank_data=bank_data[(bank_data['weekk'] >= monthRange[0]) & (bank_data['weekk'] <= monthRange[1])].copy()
    sent_bank=granular_bank_data[granular_bank_data['originn']==Account]
    recieved_bank=granular_bank_data[granular_bank_data['dest']==Account]
    tot_bank=pd.concat([sent_bank,recieved_bank])
    
    edges = pd.DataFrame({'source': tot_bank['originn'],'target':tot_bank['dest']
                          ,'weight': tot_bank['amount']
                          ,'color': ['g' if x<=200 else 'r' for x in tot_bank['amount']]
                         })

    adj_data = tot_bank

    for i in list(edges.target.unique()):
        a=granular_bank_data[granular_bank_data['originn']==i]
        b=granular_bank_data[granular_bank_data['dest']==i]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
        
    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })

    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    got_net=net.Network(height="750px", width="100%", bgcolor="white", font_color="black",directed=True,notebook=True)
    got_net.from_nx(G_two_edge)
    got_net.show_buttons(filter_=['physics'])
    
    got_net.save_graph("TwoEdge_net_updated.html")
    return open('TwoEdge_net_updated.html', 'r').read()


# Add the server clause:
if __name__ == '__main__':
    app.run_server()