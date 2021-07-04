import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import flask
import base64
from utils.utils import plot_pred, predict, top4
from dash_extensions import Download
from dash_extensions.snippets import send_file
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# initialize dash and flask
app = dash.Dash(__name__)
server = app.server
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/predictions/', external_stylesheets=[dbc.themes.BOOTSTRAP])#external_stylesheets)

UPLOAD_FOLDER = './static'
server.secret_key = "Petrit"
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
server.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# dash app layout
app.layout = html.Div([
    html.Div(id='dd_div'),
    html.Div(id='output-data-upload'),
    html.Img(id='image'),
    html.Img(id='image2'),
    html.Br(),
    # adding a spinner, so that the user can see that the app is working
    dcc.Loading(
            id="loading-1",
            children=html.Div(id="loading-output-1"),
            fullscreen=True,
            type='circle'
        ),
    html.Button('Start predictions!', id='pred', value='Click to start predictions', n_clicks=0,
                style={'float': 'left','margin': 'auto'}),
    html.A(html.Button('Home'),href='/', style={'float': 'left','margin': 'auto'}),
    html.Button('Get detailed view of heatmaps!', id='detailed', value='Click to start datailed run', n_clicks=0,
                style={'float': 'left', 'margin': 'auto'}),
    html.Div([html.Button("Download Image 1", id="btn1", n_clicks=0), Download(id="download1")],
             style={'float': 'left','margin': 'auto'}),
    html.Div([html.Button("Download Image 2", id="btn2", n_clicks=0), Download(id="download2")],
             style={'float': 'left','margin': 'auto'})
])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']


# delete all existing images/predictions/heatmaps
@server.route('/')
def upload_form():
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(f'static/{file}')
    for file in os.listdir('cam_pred'):
        os.remove(f'cam_pred/{file}')
    for file in os.listdir('top4'):
        os.remove(f'top4/{file}')
    return render_template('upload.html')


# save all uploaded images to the static folder
@server.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(server.config['UPLOAD_FOLDER'], filename))
        return redirect('/predictions')


# starting the predictions and returning a dropdown for the user to choose original images/heatmaps/plots of the predictions
# if the user also wants more detailed heatmaps, he gets the heatmaps of the top 4 predicted images (while the original heatmap only shows the top predicted disease)
# two ways to do this: click the detailed button and then start the predictions or click the button after the predictions have been mada
@app.callback([Output('dd_div', 'children'), Output("loading-output-1", "value")], [Input('pred', 'n_clicks'),Input('detailed', 'n_clicks') ])
def start_pred(n_clicks, n_clicks2):
    if n_clicks > 0:
        files = [file for file in os.listdir(UPLOAD_FOLDER)]
        for file in files:
            pred_list = predict(file)
            plot_pred(pred_list, file)
            if n_clicks2 > 0:
                top4(pred_list, file)
        options = [{'label': 'Raw ' + i, 'value': 'static/' + i} for i in files]
        plots = [{'label': 'Plot ' + i, 'value': 'cam_pred/plot_' + i} for i in files]
        heat = [{'label': 'Heatmap ' + i, 'value': 'cam_pred/heat_' + i} for i in files]
        if n_clicks2 > 0:
            top4_heat = [{'label': 'Heatmap_top4 ' + i, 'value': 'cam_pred/together_' + i} for i in files]
            all_files = options + plots + heat + top4_heat
        else:
            all_files = options + plots + heat
        return [html.Div([dcc.Dropdown(id='dd', options=all_files), dcc.Dropdown(id='dd2', options=all_files)])], n_clicks
    else:
        return [html.Div(id='...')],  n_clicks


# showing an image inviting to select images via the dropdown
@app.callback([Output('image', 'src'), Output('image', 'height'),
               Output('image', 'width')], Input('dd', 'value'))
def update_layout(value):
    if value is None:
        value = 'static_img/img1.png'
    encoded_image = base64.b64encode(open(value, 'rb').read())
    return [f'data:image/png;base64,{encoded_image.decode()}', '650px', '650px']

@app.callback([Output('image2', 'src'), Output('image2', 'height'),
               Output('image2', 'width')], Input('dd2', 'value'))
def update_layout(value):
    if value is None:
        value = 'static_img/img2.png'
    encoded_image = base64.b64encode(open(value, 'rb').read())
    return [f'data:image/png;base64,{encoded_image.decode()}', '650px', '650px']


# option for the user to download the currently displayed images
@app.callback(Output("download1", "data"), [Input("btn1", "n_clicks"), Input('dd', 'value')])
def down1(n_clicks, value):
    if n_clicks > 0:
        if value is None:
            value = 'static_img/img1.png'
        return send_file(value)

@app.callback(Output("download2", "data"), [Input("btn2", "n_clicks"), Input('dd2', 'value')])
def down2(n_clicks, value):
    if n_clicks > 0:
        if value is None:
            value = 'static_img/img2.png'
        return send_file(value)


@server.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    server.run(debug=True)
