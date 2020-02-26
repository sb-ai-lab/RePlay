"""
Дашборд, рисует дашборд.
в параметрах запуска нужно указать путь до csv с результатами и полями model_name и метриками
``python app.py ../../res.csv``
"""
import argparse

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
from dash_table.Format import Format, Scheme

external_stylesheets = ['style.css']

parser = argparse.ArgumentParser(description='Launch DashBoard')
parser.add_argument("path", help='path to results csv')
args = parser.parse_args()

df = pd.read_csv(args.path)

metrics = df.columns[df.columns != 'model_name']
dff = pd.melt(df, id_vars=['model_name'], value_vars=metrics)
dff = dff.rename(columns={"variable": "metric"})

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = px.bar(
    dff,
    x="metric",
    y="value",
    color='model_name',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Prism
)
app.layout = html.Div(children=[
    html.H1('Dashboard'),

    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{
            'id': c,
            'name': c,
            'format': Format(
                precision=4,
                scheme=Scheme.fixed,
            ),
            "type": "numeric"
        } for c in df.columns],
        sort_action='native',
        style_data_conditional=[
            {
                'if': {
                    'column_id': metric,
                    'filter_query': f'{{{metric}}} >= {df[metric].max()}'
                },
                'backgroundColor': '#3D9970',
                'color': 'white'
            } for metric in metrics
        ]
    ),

    dcc.Graph(
        id='metrics',
        figure=fig
    )
],
    style={'width': '50%', 'margin': 'auto'})

if __name__ == '__main__':
    app.run_server(debug=True)
