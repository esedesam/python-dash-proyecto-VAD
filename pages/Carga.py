import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import io
import plotly.graph_objs as go
import datetime
import base64

# Crear la aplicación Dash
dash.register_page(__name__, path='/')

layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        filename = None,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(
        id='output-data-upload',
        children=dash_table.DataTable(
            id='dataTableDisplay'))
])

def parse_contents(contents, fileName, date, storeData):
    message = None
    if not fileName is None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in fileName:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            if fileName == 'Arritmias.csv':
                cols = df.columns[1:-4]
                for col in cols:
                    df[col] = df[col].str.replace(",", ".").astype(float)
                message = html.Div("Hola, viejo conocido mío. Conozco tu preprocesamiento.")
        elif 'xls' in fileName:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise PreventUpdate
        
        storeData['data'] = df.to_dict()
        storeData['metaData'] = {
            'date': date,
            'fileName': fileName}
        
    if not ('data' in storeData and 'metaData' in storeData):
        raise PreventUpdate

    df = pd.DataFrame.from_dict(storeData['data'])
    # Mostrar solo las primeras 10 filas del DataFrame
    df_head = df.head(10)

    tableDiv = html.Div([
        html.H5(storeData['metaData']['fileName']),
        html.H6(datetime.datetime.fromtimestamp(storeData['metaData']['date']).strftime('%Y-%m-%d %H:%M:%S')),  # Formatear datetime como cadena

        message,  # Agregar el mensaje aquí
        html.Hr(),
        
        # Configuración de la tabla interactiva
        dash_table.DataTable(
            id = 'dataTableDisplay',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df_head.columns],
            style_table={'width': '80%', 'margin': 'auto'},
            editable=True
        ),

        html.Hr()  # horizontal line
    ])
    return tableDiv, storeData

@callback(
        Output('output-data-upload', 'children'),
        Output('store', 'data'), # con esto se cambia el contenido almacenado en el dcc.Store()
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified'),
        Input('store', 'data'),
        Input('dataTableDisplay', 'data'),
        State('output-data-upload', 'children'),
        prevent_initial_call=True)
def update_output(content, fileName, date, storeData, rows, tableDiv):
    triggeredId = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggeredId in ['upload-data', 'store']:
        tableDiv, storeData = parse_contents(content, fileName, date, storeData)
    elif triggeredId in ['dataTableDisplay']:
        df = pd.DataFrame(rows)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        storeData['data'] = df.to_dict()
    else:
        raise PreventUpdate
    return tableDiv, storeData

