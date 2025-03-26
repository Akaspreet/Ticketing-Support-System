"""
Dashboard module for displaying LangSmith runs status.
"""

import dash     #pylint: disable=import-error
import pandas as pd
from dash import dcc, html, Input, Output           #pylint: disable=import-error
import plotly.express as px                         #pylint: disable=import-error

from app.api.services.langsmith_service import langsmith_service        #pylint: disable=import-error

# Initialize the Dash app
dash_app = dash.Dash(
    __name__,
    server=True,
    url_base_pathname='/dashboard/'
)

# Define the layout of the Dash app
dash_app.layout = html.Div([
    html.H1('LangSmith Local Dashboard'),
    dcc.Dropdown(
        id='status-filter',
        options=[
            {'label': 'All', 'value': 'all'},
            {'label': 'Completed', 'value': 'completed'},
            {'label': 'Failed', 'value': 'failed'}
        ],
        value='all',
        clearable=False
    ),
    dcc.Graph(id='runs-bar-chart'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute
        n_intervals=0
    )
])

# Define callback to update the bar chart based on the filter
@dash_app.callback(
    Output('runs-bar-chart', 'figure'),
    [Input('status-filter', 'value'), Input('interval-component', 'n_intervals')]
)
def update_bar_chart(selected_status, _):
    """
    Update the bar chart based on the selected status filter.

    Args:
        selected_status (str): The selected status filter.
        _ (int): Interval component (unused).

    Returns:
        plotly.graph_objs._figure.Figure: The bar chart figure.
    """
    df = pd.DataFrame(list(langsmith_service.runs_collection.find()))
    if selected_status != 'all':
        df = df[df['status'] == selected_status]
    fig = px.bar(df, x='name', color='status', title='Runs by Status')
    return fig
