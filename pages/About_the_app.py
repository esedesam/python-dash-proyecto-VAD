import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Registrar la página y establecer su orden
dash.register_page(__name__, order=4)

# Definir el layout de la aplicación
layout = dbc.Container([
    html.Div([
        html.H1("Acerca de la Aplicación"),
        html.P("La aplicación 'Dataset Inspector' ha sido desarrollada por Jesús Martínez Leal como parte de un proyecto de la asignatura de 'Visualización Avanzada de Datos' del Máster en Ciencia de Datos de la Universitat de València."),
        html.P("El propósito de esta aplicación es proporcionar una herramienta interactiva para visualizar y explorar datos de manera efectiva, concretando en el dataset de Arritmias.csv proporcionado por el profesor."),
        html.Span("Para ello, se hizo uso de Dash + Plotly. Dash es un framework en Python para construir aplicaciones web interactivas, mientras que Plotly es una biblioteca de visualización de datos que permite crear gráficos interactivos y visualizaciones de datos dinámicas. Juntos, Dash y Plotly ofrecen una forma poderosa y flexible de crear aplicaciones web interactivas con capacidades de visualización de datos avanzadas."),
        html.Hr(),
        html.H2("Sobre el autor"),
        html.P("El autor de esta aplicación es:"),
        html.Ul([
            html.Li([html.A("Martínez Leal, Jesús", href="mailto:jemarle@alumni.uv.es")])
        ])
    ])
], fluid=True)
