"""Enhanced PatchCore Dashboard (scaffold)

This app uses dash-bootstrap-components for a responsive layout.
It reuses helper functions from the project's `patchcore_dashboard.py` where available.
"""
import sys
from pathlib import Path as _Path

# import helpers from the local functions package
HERE = _Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from functions.helpers import array_to_base64, calculate_anomaly_map, draw_smiley

# tiny usage to avoid linter unused-import warnings (real usage will follow)
_helpers_used = (array_to_base64, draw_smiley)

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL
import plotly.graph_objects as go
import numpy as np
from functions.dataset import generate_demo_dataset


# create demo dataset
DATASET = generate_demo_dataset()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


def make_sidebar():
    return dbc.Card(
        dbc.CardBody([
            html.H5("Controls", className="card-title"),
            dbc.Label("Patch size"),
            dcc.Slider(id="patch-size", min=32, max=256, step=8, value=64),
            html.Br(),
            dbc.Label("Stride"),
            dcc.Slider(id="stride", min=1, max=32, step=1, value=8),
            html.Hr(),
            dbc.Button("Reload dataset", id="reload-btn", color="primary", className="mb-2"),
            html.Div(id="dataset-info"),
        ])
    )


def make_thumbnail_grid():
    # placeholder thumbnails grid
    thumbs = []
    for i, t in enumerate(DATASET['thumbs']):
        thumb_src = array_to_base64(t)
        thumbs.append(dbc.Col(html.Img(src=thumb_src, id={'type': 'thumb', 'index': i}, style={'width': '32px', 'height': '32px', 'objectFit': 'cover', 'cursor': 'pointer', 'margin': '2px'}), width="auto"))

    return dbc.Row(thumbs, className="g-2")


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("PatchCore Enhanced Dashboard"), width=8),
        dbc.Col(html.Div("Status: ready", id="status"), width=4),
    ], align="center", className="my-2"),

    dbc.Row([
        dbc.Col(make_sidebar(), width=3),
        dbc.Col(dcc.Loading(id="main-loading", children=[
            dcc.Graph(id="main-heatmap", style={'height': '70vh'})
        ]), width=6),
        dbc.Col(html.Div([html.H6("Thumbnails"), make_thumbnail_grid()]), width=3),
    ]),

    dcc.Store(id="selected-image", data={}),
    dcc.Store(id="dataset-store", data=DATASET),
], fluid=True)


@app.callback(Output('selected-image', 'data'), Input({'type': 'thumb', 'index': ALL}, 'n_clicks'), State('selected-image', 'data'))
def thumbnail_click(n_clicks_list, current):
    # find which thumb was clicked (the first non-None click)
    if not n_clicks_list:
        return dash.no_update
    for i, v in enumerate(n_clicks_list):
        if v:
            return {'index': i}
    return current


@app.callback(Output('dataset-store', 'data'), Input('reload-btn', 'n_clicks'))
def reload_dataset(n_clicks):
    # regenerate dataset on demand
    if not n_clicks:
        return dash.no_update
    ds = generate_demo_dataset()
    return ds


@app.callback(Output('main-heatmap', 'figure'), Input('selected-image', 'data'), Input('patch-size', 'value'), Input('stride', 'value'))
def update_main(selected, patch_size, stride):
    # select image from dataset (default to first selectable)
    idx = 0
    try:
        if selected and 'index' in selected:
            idx = int(selected['index'])
        else:
            idx = DATASET['selectable_indices'][0]
    except Exception:
        idx = 0
            
    # read current dataset (if replaced by reload)
    ds = DATASET
    try:
        store = dash.callback_context.states['dataset-store.data']
        if store:
            ds = store
    except Exception:
        pass

    img = ds['images'][idx]
    # compute anomaly map
    amap = calculate_anomaly_map(patch_size, stride, img)

    # upsample anomaly map to image size
    try:
        scale_y = max(1, img.shape[0] // amap.shape[0])
        scale_x = max(1, img.shape[1] // amap.shape[1])
        amap_upsampled = np.kron(amap, np.ones((scale_y, scale_x)))
        amap_upsampled = amap_upsampled[:img.shape[0], :img.shape[1]]
    except Exception:
        amap_upsampled = np.zeros((img.shape[0], img.shape[1]))

    # build a subplot with two panels: image | anomaly map
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image", "Anomaly map"))
    fig.add_trace(go.Image(z=img), row=1, col=1)
    fig.add_trace(go.Heatmap(z=amap_upsampled, colorscale='RdYlBu', showscale=True, zmin=0.0, zmax=1.0), row=1, col=2)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
    return fig


@app.callback(Output('dataset-info', 'children'), Input('selected-image', 'data'))
def update_info(selected):
    try:
        idx = int(selected.get('index', DATASET['selectable_indices'][0]))
    except Exception:
        idx = DATASET['selectable_indices'][0]
    label = DATASET['labels'][idx]
    return html.Div([html.Strong(f"Index: {idx}"), html.Br(), html.Span(f"Label: {label}")])


if __name__ == '__main__':
    # When launching in background or from a non-interactive shell, the
    # Werkzeug reloader can attempt to access the TTY and fail with
    # termios.error: Interrupted system call. Disable the reloader when
    # running detached.
    app.run(debug=True, port=8052, use_reloader=False)
