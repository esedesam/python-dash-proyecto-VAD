import dash
from dash import dcc, html, Input, Output, callback,  State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
from dash.exceptions import PreventUpdate
from sklearn.manifold import TSNE
from io import StringIO
import sys

dash.register_page(__name__)

rs = 0
n_components = 2
n_iter = 10000
n_iter_without_progress = 300
verbose = 1
method = 'exact'
init = 'pca'
early_exaggeration = 25

# Definir el layout de la aplicación
layout = dbc.Container([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column2',
                style={"margin-bottom": "5px"})],
            style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column2',
                style={"margin-bottom": "5px"})],
            style={'width': '49%', 'float': 'right', 'display': 'inline-block'})],
        style={
            'borderBottom': 'thin lightgrey solid',
            'backgroundColor': 'rgb(250, 250, 250)',
            'padding': '5px 10px'}),
    html.Div([
    html.Label('Perplexity:'),
    dcc.Input(id='perplexity-input', type='number', value=30, min=5, max=50, step=1),
    html.Label('Learning Rate:'),
    dcc.Input(id='learning-rate-input', type='number', value=200, min=10, max=1000, step=10),
    html.Button('Update', id='update-button', n_clicks=0)
    ], style={'margin-top': '20px'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter2',
            hoverData={'points': [{'hovertext': 'P1'}]})],
        style={'display': 'inline-block', 'width': '49%', 'padding': '5px 5px'}),
    html.Div([
        dcc.Graph(id='x-hist2'),
        dcc.Graph(id='y-hist2')],
        style={'display': 'inline-block', 'width': '49%', 'padding': '5px 5px'})
],
    fluid=True
)

@callback(
    Output('crossfilter-xaxis-column2', 'options'),
    Output('crossfilter-xaxis-column2', 'value'),
    Output('crossfilter-yaxis-column2', 'options'),
    Output('crossfilter-yaxis-column2', 'value'),
    Input('store', 'data'))
def updateColDropdown(storeData):
    if 'data' in storeData and not storeData['data'] == {}:
        newOptions = list(storeData['data'].keys())[1:]
    else:
        newOptions = []
    return newOptions, newOptions[0], newOptions, newOptions[1]

@callback(
    Output('crossfilter-indicator-scatter2', 'figure'),
    Input('crossfilter-xaxis-column2', 'value'),
    Input('crossfilter-yaxis-column2', 'value'),
    Input('store', 'data'),
    Input('update-button', 'n_clicks'),
    State('perplexity-input', 'value'),
    State('learning-rate-input', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name, storeData, n_clicks, perplexity_value, learning_rate_value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'update-button' in changed_id:
        if n_clicks > 0:
            return generate_tsne_scatter(storeData, perplexity_value, learning_rate_value)
        else:
            raise PreventUpdate
    else:
        if 'data' in storeData and not storeData['data'] == {}:
            return generate_tsne_scatter(storeData)
        else:
            return px.scatter()

def generate_tsne_scatter(storeData, perplexity_value=30, learning_rate_value=200):
    df = pd.DataFrame.from_dict(storeData['data'])
    df_tsne = perform_tsne(df, perplexity_value, learning_rate_value)
    df_tsne['AV'] = df_tsne['AV'].astype(str)
    
    fig = px.scatter(
        df_tsne,
        x='t-SNE Component 1',
        y='t-SNE Component 2',
        height=500,
        template="ggplot2",
        hover_name=df[df.columns[0]],
        color='AV')
    return fig

def perform_tsne(df, perplexity_value=30, learning_rate_value=200):
    tsne = TSNE(n_components=n_components, random_state=rs, perplexity=perplexity_value, n_iter=n_iter,
                learning_rate=learning_rate_value, verbose=verbose,
                n_iter_without_progress=n_iter_without_progress, method=method, init=init,
                early_exaggeration=early_exaggeration)

    df_dropped = df.drop([df.columns[0], df.columns[-1]], axis=1)
    tsne_result = tsne.fit_transform(df_dropped)
    tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE Component 1', 't-SNE Component 2'])
    tsne_df['AV'] = df['AV'].values
    return tsne_df

def create_hist(df, xaxis_column_name, nBins=10, patient=None, patientsColName='PACIENTES'):
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

    if not patient is None:
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

@callback(
    Output('x-hist2', 'figure'),
    Input('crossfilter-indicator-scatter2', 'hoverData'),
    Input('crossfilter-xaxis-column2', 'value'),
    Input('store', 'data'))
def update_x_hist(hoverData, xaxis_column_name, storeData):
    if hoverData is None:
        raise PreventUpdate
    patient = hoverData['points'][0]['hovertext']
    df = pd.DataFrame.from_dict(storeData['data'])
    fig = create_hist(df, xaxis_column_name, patient=patient)
    return fig

@callback(
    Output('y-hist2', 'figure'),
    Input('crossfilter-indicator-scatter2', 'hoverData'),
    Input('crossfilter-yaxis-column2', 'value'),
    Input('store', 'data'))
def update_y_hist(hoverData, yaxis_column_name, storeData):
    if hoverData is None:
        raise PreventUpdate
    patient = hoverData['points'][0]['hovertext']
    df = pd.DataFrame.from_dict(storeData['data'])
    fig = create_hist(df, yaxis_column_name, patient=patient)
    return fig
