import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback

dash.register_page(__name__)

df = pd.read_csv('./data/Arritmias.csv')
available_indicators = df.columns[1:]

cols = df.columns[1:-4]
for i in range(len(cols)):
      df[cols[i]] = df[cols[i]].str.replace(",", ".").astype(float)
      
# Definir el layout de la aplicación
layout = dbc.Container([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                # options=[{'label': i, 'value': i} for i in available_indicators],
                # value=available_indicators[0],
                style={"margin-bottom": "5px"}),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'})],
            style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                # options=[{'label': i, 'value': i} for i in available_indicators],
                # value=available_indicators[1],
                style={"margin-bottom": "5px"}),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'})],
            style={'width': '49%', 'float': 'right', 'display': 'inline-block'})],
        style={
            'borderBottom': 'thin lightgrey solid',
            'backgroundColor': 'rgb(250, 250, 250)',
            'padding': '5px 10px'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'hovertext': 'P1'}]})],
        style={'display': 'inline-block', 'width': '49%', 'padding': '5px 5px'}),
    html.Div([
        dcc.Graph(id='x-hist'),
        dcc.Graph(id='y-hist')],
        style={'display': 'inline-block', 'width': '49%', 'padding': '5px 5px'}),
    ]
    ,fluid = True
)

# Interacción de store-data -> rehacer dropdowns
@callback(
        [Output('crossfilter-xaxis-column', 'options'),
         Output('crossfilter-xaxis-column', 'value'),
         Output('crossfilter-yaxis-column', 'options'),
         Output('crossfilter-yaxis-column', 'value')],
        Input('store', 'data'))
def updateColDropdown(store):
    if store == {}:
        newOptions = []
    else:
        newOptions = list(store.keys())[1:]
    print(f'Nuevas opciones: {newOptions}')
    return newOptions, newOptions[0], newOptions, newOptions[1]

# Interacción de menús y slider -> rehacer scatter
@callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('crossfilter-xaxis-type', 'value'),
     Input('crossfilter-yaxis-type', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type):
    fig = px.scatter(
        df,
        x=xaxis_column_name,
        y=yaxis_column_name,
        log_x=(xaxis_type == 'Log'),
        log_y=(yaxis_type == 'Log'),
        height=500,
        template="ggplot2",
        hover_name='PACIENTES',
        hover_data=df.columns[1:].tolist())
    return fig

# Función para crear histograma
def create_hist(df, xaxis_column_name, axis_type, nBins=10, patient='', patientsColName='PACIENTES'):
    binValues, binEdges = np.histogram(df[xaxis_column_name].values, bins=nBins)
    fig = px.histogram(
        df,
        x=xaxis_column_name,
        nbins=nBins,
        hover_data=df.columns,
        height=250,
        template="ggplot2")
    fig.update_traces(xbins={
        'start': binEdges[0],
        'end': binEdges[-1],
        'size': binEdges[1] - binEdges[0]})
    fig.update_xaxes(range=(binEdges[0], binEdges[-1]))

    pacientXValue = df[df[patientsColName] == patient][xaxis_column_name].values[0]
    endEdgeIdx = np.searchsorted(binEdges, pacientXValue)
    if endEdgeIdx == 0:
        endEdgeIdx = 1
    startBinEdge, endBinEdge = binEdges[endEdgeIdx - 1], binEdges[endEdgeIdx]
    binHeight = binValues[endEdgeIdx - 1]
    fig.add_shape(
        type='rect',
        x0=startBinEdge,
        y0=0,
        x1=endBinEdge,
        y1=binHeight,
        line={'width': 1},
        fillcolor='red',
        opacity=1)
    fig.update_layout(
        margin={'l': 5, 'r': 5, 't': 20, 'b': 5})
    return fig

# Interacción de hoverData o menús -> rehacer series 
@callback(
    Output('x-hist', 'figure'),
    [Input('crossfilter-indicator-scatter', 'hoverData'),
     Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-xaxis-type', 'value')])
def update_x_hist(hoverData, xaxis_column_name, axis_type):
    patient = hoverData['points'][0]['hovertext']
    fig = create_hist(df, xaxis_column_name, axis_type, patient=patient)
    return fig

@callback(
    Output('y-hist', 'figure'),
    [Input('crossfilter-indicator-scatter', 'hoverData'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('crossfilter-yaxis-type', 'value')])
def update_y_hist(hoverData, yaxis_column_name, axis_type):
    patient = hoverData['points'][0]['hovertext']
    fig = create_hist(df, yaxis_column_name, axis_type, patient=patient)
    return fig

