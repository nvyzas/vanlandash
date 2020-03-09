#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LOOK INTO OVERWRITING CSS https://stackoverflow.com/questions/50844844/python-dash-custom-css
# GOAL IS TO OVERWRITE SOME 2PX DEFAULTS IN THIS FILE https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis.css

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

import networkx as nx
from pyvis import network as net


### Data import and preprocessing

df=pd.read_csv('data/threeMonths.csv')

df['datee'] = pd.to_datetime(df['datee']).dt.date

print('Finding unique accounts')
unique_accounts=df['a_key'].append(df['b_key'],ignore_index=True).unique().tolist()
print('Done')

unique_accounts.sort()
min_date=df['datee'].min()
max_date=df['datee'].max()

### Utility functions

def filter_df(acc_key, start_date=min_date, end_date=max_date):
    condition=(df['datee']>=start_date) & (df['datee']<=end_date) & ((df['a_key']==acc_key) | (df['b_key']==acc_key))
    return df[condition]

### Network similarity functions

def _is_close(d1, d2, atolerance=0, rtolerance=0):
    # Pre-condition: d1 and d2 have the same keys at each level if they are dictionaries
    if not isinstance(d1, dict) and not isinstance(d2, dict):
        return abs(d1 - d2) <= atolerance + rtolerance * abs(d2)
    return all(all(_is_close(d1[u][v], d2[u][v]) for v in d1[u]) for u in d1)

def unique(list1): 
    
    # intilize a null list 
    unique_list = [] 
        
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return  unique_list

def precssr(list1):
    res=[x[0] for x in unique(list1)]
    return res
        
def simrank_similarity_Incoming(G, source=None, target=None, importance_factor=0.9, max_iterations=100, tolerance=1e-4):
    prevsim = None
        # build up our similarity adjacency dictionary output
    newsim = {u: {v: 1 if u == v else 0 for v in G} for u in G}
        # These functions compute the update to the similarity value of the nodes
        # `u` and `v` with respect to the previous similarity values.
    avg_sim = lambda s: sum(newsim[w][x] for (w, x) in s) / len(s) if s else 0.0
    sim = lambda u, v: importance_factor * avg_sim(list(product(
    precssr(list(G.in_edges(u, data=False))),
    precssr(list(G.in_edges(v, data=False)))
    )))
    for _ in range(max_iterations):
        if prevsim and _is_close(prevsim, newsim, tolerance):
            break
        prevsim = newsim
        newsim = {u: {v: sim(u, v) if u is not v else 1
                    for v in newsim[u]} for u in newsim}
    if source is not None and target is not None:
        return newsim[source][target]
    if source is not None:
        return newsim[source]
    return newsim

def sim_outgoing_two_nodes(sim_dict,a,b):
    return sim_dict[a][b]

def sim_incoming_two_nodes(sim_dict,a,b):
    return sim_dict[a][b]

### App Layout
    
app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout=html.Div([
        html.Div(
            className="col-4",
            id='heading',
            children=[
                html.H1(children='Van Lanschot Bank', 
                style={
                'textAlign': 'right',
                'color': colors['text']
                })
            ]
        ),
        html.Div(
            className="row",
            style={'height': '46px',
                   'margin-top': '4px',
                   'margin-bottom': '4px'},
            children=[
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Input(
                         id='input_1',
                         type='text',
                         value=unique_accounts[7],
                         style={'height': '47px'}
                         )
                    ]                        
                ),
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Input(
                         id='input_2',
                         type='text',
                         style={'height': '47px'}
                         )
                    ]                        
                ),
                html.Div(
                    className="col-4",
                    children=[
                        dcc.DatePickerRange(
                            id='datepicker',
                            display_format='DD MM YYYY',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation='horizontal'
                        )  
                    ]
                ),
                html.Div(
                    className="col-3",
                    children=[
                        dcc.RangeSlider(
                            id='amount-slider',
                            step=1, # todo: set this in callback according to slider value
                        ),
                        html.Div(id='amount-slider-output')
                    ]
                ),
                html.Div( # Probably need to change this to children / input / text
                    className="col-1",
                    id='textArea',
                    children=[  
                        dcc.Textarea( # todo: change this into DIV
                            placeholder='SimRank',
                            #value='Similarity',
                            style={'width': '100%'#,
                            #'height': '69%'
                            }
                        )
                    ]
                ),
                html.Div(
                    className="col-1",
                    children=[
                        html.Button(
                        id='submit-button',
                        children='Submit',
                        n_clicks=0
                        )
                    ])
            ]
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className='col-6',
                    children=[
                        html.Iframe( # accessKey, aria-*, children, className, contentEditable, contextMenu, data-*, dir, draggable, height, hidden, id, key, lang, loading_state, n_clicks, n_clicks_timestamp, name, role, sandbox, spellCheck, src, srcDoc, style, tabIndex, title, width
                            id='mapx', 
                            width='100%', 
                            height='100%'
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
        )
])


### Callbacks

# Update key_2 based on key_1
@app.callback(
    Output('input_2','value'),
    [Input('input_1','value')])
def update_key_2(key_1):
    print('Updating key 2')
    
    df_filtered_1=filter_df(key_1)
    unique_accounts_filtered=df_filtered_1['a_key'].append(df_filtered_1['b_key'],ignore_index=True).unique()
    unique_accounts_filtered_without_key_1=np.delete(unique_accounts_filtered,np.where(unique_accounts_filtered==key_1))    
    key_2 = unique_accounts_filtered_without_key_1[0]
    
    print('Done updating key 2')
    return key_2

# Update dates and amounts based on key_1 and key_2
@app.callback(
    [Output('datepicker','min_date_allowed'),
     Output('datepicker','max_date_allowed'),
     Output('datepicker','start_date'),
     Output('datepicker','end_date'),
     Output('datepicker','initial_visible_month'),
     Output('amount-slider','min'),
     Output('amount-slider','max'),
     Output('amount-slider','value')],
    [Input('input_1','value'),
     Input('input_2','value')])
def update_dates_and_amounts(key_1,key_2):
    print('Updating dates and amounts')
    
    if (key_1==None) or (key_2==None):
        print("Preventing update of dates and amounts")
        raise PreventUpdate
        
    # filter global df according to key_1 and key_2
    
    df_filtered_1=filter_df(key_1)
    df_filtered_2=filter_df(key_2)
    
    # calculate min and max dates
    
    min_date_1=df_filtered_1['datee'].min()
    max_date_1=df_filtered_1['datee'].max()
    
    min_date_2=df_filtered_2['datee'].min()
    max_date_2=df_filtered_2['datee'].max()
    
    min_date_12=min_date_1 if (min_date_1<=min_date_2) else min_date_2
    max_date_12=max_date_1 if (max_date_1>=max_date_2) else max_date_2
    
    # calculate min and max absolute amounts
    
    min_amount_1=df_filtered_1['boeking_eur'].abs().min()
    max_amount_1=df_filtered_1['boeking_eur'].abs().max()
    
    min_amount_2=df_filtered_2['boeking_eur'].abs().min()
    max_amount_2=df_filtered_2['boeking_eur'].abs().max()
    
    min_amount_12=min_amount_1 if min_amount_1<=min_amount_2 else min_amount_2
    max_amount_12=max_amount_1 if max_amount_1>=max_amount_2 else max_amount_2
    
    print('Done updating dates and amounts')
    return min_date_12,max_date_12,min_date_12,min_date_12,min_date_12,min_amount_12,max_amount_12,[min_amount_12,max_amount_12]
     
# Update time and frequency graphs based on all inputs
@app.callback(
    [Output('timegraph','figure'),
     Output('freqgraph','figure')],
    [Input('submit-button', 'n_clicks')],
    [State('input_1','value'),
     State('input_2','value'),
     State('datepicker','start_date'),
     State('datepicker','end_date')])
def update_time_and_frequency_graphs(n_clicks,key,key_2,start_date,end_date):
    print('Updating graphs')
    
    if ((key is None) or (key_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of time and frequency graphs')
        raise PreventUpdate
        
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
        # width=800,
        # height=800,
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
        # width=800,
        # height=800,
        margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
        hovermode='closest'
    )
    
    print('Done updating graphs')       
    return fig,fig_freq

# Update network based on all inputs
@app.callback(
    Output('mapx','srcDoc'), 
    [Input('submit-button','n_clicks')],
    [State('input_1','value'),
     State('input_2','value'),
     State('datepicker','start_date'),
     State('datepicker','end_date')])
def update_network(n_clicks,Account_1,Account_2,start_date,end_date):
    
    if ((Account_1 is None) or (Account_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of time and frequency graphs')
        raise PreventUpdate
        
    bank_data = df.copy()
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    granular_bank_data=bank_data[(bank_data['datee'] >= sd) & (bank_data['datee'] <= ed)]
    sent_bank=granular_bank_data[granular_bank_data['originn']==Account_1]
    recieved_bank=granular_bank_data[granular_bank_data['dest']==Account_1]
    tot_bank=pd.concat([sent_bank,recieved_bank])
    
    edges = pd.DataFrame({'source': tot_bank['originn'],'target':tot_bank['dest']
                          ,'weight': tot_bank['amount']
                          ,'color': ['g' if x<=200 else 'r' for x in tot_bank['amount']]
                         })
    adj_data = tot_bank
    if Account_1 != Account_2:
        a=granular_bank_data[granular_bank_data['originn']==Account_2]
        b=granular_bank_data[granular_bank_data['dest']==Account_2]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
    
    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })
     
    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    output_filename='TwoEdge_net_updated.html'
    # make a pyvis network
    network_class_parameters = {"notebook": True, "height": "98vh", "width":"98vw", "bgcolor": None,"font_color": None, "border": 0, "margin": 0, "padding": 0} # 
    pyvis_graph = net.Network(**{parameter_name: parameter_value for parameter_name,
                                 parameter_value in network_class_parameters.items() if parameter_value}, directed=True) 
    sources = two_edges['source']
    targets = two_edges['target']
    weights = two_edges['weight']
    color = two_edges['color']
    edge_data = zip(sources, targets, weights, color)
    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
        c = e[3]
        pyvis_graph.add_node(src,title=src)
        pyvis_graph.add_node(dst,title=dst)
        pyvis_graph.add_edge(src,dst,value=w,color=c)   
    #pyvis_graph.show_buttons(filter_=['nodes','edges','physics'])   
    pyvis_graph.set_options("""
var options = {
  "nodes": {
    "borderWidthSelected": 3,
    "color": {
      "border": "rgba(43,124,233,1)",
      "background": "rgba(109,203,252,1)",
      "highlight": {
        "border": "rgba(55,123,233,1)",
        "background": "rgba(255,248,168,1)"
      }
    },
    "font": {
      "size": 15,
      "face": "tahoma"
    },
    "size": 17
  },
  "edges": {
    "arrowStrikethrough": false,
    "color": {
      "inherit": true
    },
    "smooth": {
      "forceDirection": "none",
      "roundness": 0.35
    }
  },
  "physics": {
    "forceAtlas2Based": {
      "springLength": 100,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "timestep": 0.49
  }
}
""")
    pyvis_graph.save_graph(output_filename)
    return open(output_filename, 'r').read()

# Update similarity of key_1 and key_2 based on outgoing and incoming edges
from itertools import product
@app.callback(Output('textArea','children'),
              [Input('submit-button','n_clicks')],
              [State('input_1','value'),
               State('input_2','value'),
               State('datepicker','start_date'),
               State('datepicker','end_date')])
def update_output_div_similarity(n_clicks,Account_1,Account_2,start_date,end_date):
    
    if ((Account_1 is None) or (Account_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of time and frequency graphs')
        raise PreventUpdate
        
    bank_data = df.copy()
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    granular_bank_data=bank_data[(bank_data['datee'] >= sd) & (bank_data['datee'] <= ed)]
    sent_bank=granular_bank_data[granular_bank_data['originn']==Account_1]
    recieved_bank=granular_bank_data[granular_bank_data['dest']==Account_1]
    tot_bank=pd.concat([sent_bank,recieved_bank])
    
    edges = pd.DataFrame({'source': tot_bank['originn'],'target':tot_bank['dest']
                          ,'weight': tot_bank['amount']
                          ,'color': ['g' if x<=200 else 'r' for x in tot_bank['amount']]
                         })
    adj_data = tot_bank
    if Account_1 != Account_2:
        a=granular_bank_data[granular_bank_data['originn']==Account_2]
        b=granular_bank_data[granular_bank_data['dest']==Account_2]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
    
    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })
    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    sim_outgoing = nx.simrank_similarity(G_two_edge)
    sim_incoming = simrank_similarity_Incoming(G_two_edge)
   
    sim1_2=str(sim_outgoing_two_nodes(sim_outgoing,Account_1,Account_2))+" "+str((sim_incoming_two_nodes(sim_incoming,Account_1,Account_2)))
    return sim1_2

# Add the server clause:
if __name__ == '__main__':
    app.run_server()