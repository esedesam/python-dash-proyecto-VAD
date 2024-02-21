import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
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
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            if filename == 'Arritmias.csv':
                # Aplicar preprocesado específico
                available_indicators = df.columns[1:]
                cols = df.columns[1:-4]
                for col in cols:
                    df[col] = df[col].str.replace(",", ".").astype(float)
                # Agregar el mensaje específico para 'Arritmias.csv'
                message = html.Div("Hola, viejo conocido mío. Conozco tu preprocesamiento.")
            else:
                message = None
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            message = None
        else:
            # If the file format is not supported, raise an exception
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(e)
        return html.Div([
            'Error: El formato de archivo no es compatible. Por favor, carga un archivo CSV o Excel.'
        ])

    # Mostrar solo las primeras 10 filas del DataFrame
    df_head = df.head(10)

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')),  # Formatear datetime como cadena

        message,  # Agregar el mensaje aquí
        html.Hr(),
        
        # Configuración de la tabla interactiva
        dash_table.DataTable(
            data=df_head.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df_head.columns],
            style_table={'width': '80%', 'margin': 'auto'},
            editable=True
        ),

        html.Hr()  # horizontal line
    ]), df.to_dict()

@callback([Output('output-data-upload', 'children'),
              Output('store', 'data')], # con esto se cambia el contenido almacenado en el dcc.Store()
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)

def update_output(content, filename, date):
    if content is not None:
        return parse_contents(content, filename, date)