import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback

dash.register_page(__name__, order = 2)

# Definir el layout de la aplicación

layout = dbc.Container([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
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
    html.Div([
    html.H6("Filtering by EDAD"),
    dcc.RangeSlider(
        id='age-slider',
        marks={i: str(i) for i in range(0, 101, 10)},
        min=0,
        max=100,
        step=1,
        value=[0, 100],
        allowCross=False
    )
], style={'width': '50%', 'float': 'left'}),
], fluid=True)

# Interacción de store-data -> rehacer dropdowns
@callback(
    Output('crossfilter-xaxis-column', 'options'),
    Output('crossfilter-xaxis-column', 'value'),
    Output('crossfilter-yaxis-column', 'options'),
    Output('crossfilter-yaxis-column', 'value'),
    Input('store', 'data'))
def updateColDropdown(storeData):
    if 'data' in storeData and not storeData['data'] == {}:
        newOptions = list(storeData['data'].keys())[1:]
    else:
        newOptions = []
    return newOptions, newOptions[0], newOptions, newOptions[1]

# Interacción de menús y slider -> rehacer scatter
@callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('store', 'data'),
    Input('age-slider', 'value'))

# actualizar gráfico según se mueve el slider de filtrado por la variable EDAD

def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, storeData, age_range):
    if 'data' in storeData and not storeData['data'] == {}:
        df = pd.DataFrame.from_dict(storeData['data'])
        # Filtrar según el rango de edad seleccionado
        df_filtered = df[(df['EDAD'] >= age_range[0]) & (df['EDAD'] <= age_range[1])]
        fig = px.scatter(
            df_filtered,
            x=xaxis_column_name,
            y=yaxis_column_name,
            log_x=(xaxis_type == 'Log'),
            log_y=(yaxis_type == 'Log'),
            height=500,
            template="ggplot2",
            hover_name='PACIENTES',
            hover_data=df.columns[1:].tolist())
        
        fig.update_layout(
        title={
            'text': f"Scatter Plot of selected variables",  # Título con la KL divergence
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    else:
        fig = px.scatter()
    return fig

# Función para crear histograma
def create_hist(df, xaxis_column_name, axis_type, age_range, hoverData, nBins=10, patientsColName='PACIENTES'):
    df = df[(df['EDAD'] >= age_range[0]) & (df['EDAD'] <= age_range[1])]
    fig = px.histogram(
        df,
        x=xaxis_column_name,
        nbins=nBins,
        hover_data=df.columns,
        height=250,
        template="ggplot2")
    
    # Descomentar para resaltar en el histograma (junto con lo de un poco más abajo)
    # binValues, binEdges = np.histogram(df[xaxis_column_name].values, bins=nBins)
    # fig.update_traces(xbins={
    #     'start': binEdges[0],
    #     'end': binEdges[-1] * 1.01, # con este factor multiplicativo aparece alguna cosa que estaba al límite, aunque no sale del todo bien las cosas y no creo que sea general para cualquier variable
    #     'size': binEdges[1] - binEdges[0]})
    # max_x_value = binEdges[-1] * 1.01    
    # fig.update_xaxes(range=(binEdges[0], max_x_value))

    try:
        patient = hoverData['points'][0]['hovertext']
        pacientXValue = df[df[patientsColName] == patient][xaxis_column_name].values[0]
    except:
        patient = None
    else:
        
        # Descomentar para resaltar en el histograma
        # endEdgeIdx = np.searchsorted(binEdges, pacientXValue)
        # if endEdgeIdx == 0:
        #     endEdgeIdx = 1
        # startBinEdge, endBinEdge = binEdges[endEdgeIdx - 1], binEdges[endEdgeIdx]
        # binHeight = binValues[endEdgeIdx - 1]
        # fig.add_shape(
        #     type='rect',
        #     x0=startBinEdge,
        #     y0=0,
        #     x1=endBinEdge,
        #     y1=binHeight,
        #     line={'width': 1},
        #     fillcolor='red',
        #     opacity=1)
        fig.add_vline(
            x = pacientXValue,
            line_width = 1,
            line_dash = 'dash',
            opacity = 1,
            label={
                'text': patient,
                'textposition': 'end',
                'font': {
                    'size': 18,
                    'color': 'black'},
                'yanchor': 'top',
                'xanchor': 'left',
                'textangle': 0})
    fig.update_layout(
        margin={'l': 5, 'r': 5, 't': 20, 'b': 5})
    
    return fig

# Interacción de hoverData o menús -> rehacer series 

@callback(
    Output('x-hist', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('store', 'data'),
    Input('age-slider', 'value'))
def update_x_hist(hoverData, xaxis_column_name, axis_type, storeData, age_range):
    df = pd.DataFrame.from_dict(storeData['data'])
    fig = create_hist(df, xaxis_column_name, axis_type, age_range, hoverData)
    return fig

@callback(
    Output('y-hist', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('store', 'data'),
    Input('age-slider', 'value'))
def update_y_hist(hoverData, yaxis_column_name, axis_type, storeData, age_range):
    df = pd.DataFrame.from_dict(storeData['data'])
    fig = create_hist(df, yaxis_column_name, axis_type, age_range, hoverData)
    return fig