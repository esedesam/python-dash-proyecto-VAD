import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# 2 - Data loading and app creation

#BS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"

# good options: JOURNAL, SIMPLEX, 

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SIMPLEX, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

# 3 - Creating the HTML sections with their components (dcc)


iconos = ["fa-solid fa-upload", "fa-solid fa-minimize", "fa-sharp fa-solid fa-chart-line", "fa-regular fa-face-smile"]


navbar = dbc.NavbarSimple(
    [
        dbc.Row(
            [
                dbc.Col(html.I(className=iconos[i]), width=2), # valores ajustados manualmente para que quede bien
                dbc.Col(dbc.NavLink(page["name"], href=page["path"]), width=1)
            ],
            align="center",
        )
        for i, page in enumerate(dash.page_registry.values())
        if page["module"] != "pages.not_found_404"
    ],
    brand="Dataset Inspector",
    color="primary",
    dark=True,
    className="mb-2",
)


app.layout = html.Div([
    navbar,
    dash.page_container,
    dcc.Store(id='store', data={})])

if __name__ == '__main__':
    app.run_server(debug=True)
