import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.DataFrame({"model_name": ["NeuMF", "CML", "LRML"],
                   "nDCG@10": [0.4450, 0.5413, 0.5453],
                   "HR@10": [0.7260, 0.7216, 0.7397]})

metrics = df.columns[df.columns != 'model_name']
dff = pd.melt(df, id_vars=['model_name'], value_vars=metrics)
dff = dff.rename(columns={"variable": "metric"})

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
import plotly.express as px
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
        columns=[{'id': c, 'name': c} for c in df.columns],
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