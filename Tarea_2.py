# 1 - Importamos los modulos necesarios
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc # multipág
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# 2 - Carga de datos y creacion de la app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions = True)

# print(dash.page_registry.keys())


# 3 - Creamos las secciones del html con sus componentes (dcc)

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav = True,
        label = "More Pages",
    ),
    brand="Dataset Inspector",
    color="primary",
    dark=True,
    className="mb-2",
)

app.layout = html.Div([
    navbar,
    dash.page_container,
    dcc.Store(id = 'store', data = {})])

if __name__ == '__main__':
    app.run_server(debug = True)
