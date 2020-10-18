import dash_bootstrap_components as dbc
import flask
from flask import request, send_file
import dash
from flask import send_file
import io
from base64 import encodebytes
from PIL import Image
from flask import abort, redirect, url_for
from sqlalchemy import create_engine

import cv2
engine = create_engine('sqlite:///sqllite.db')
from Model.nnAPI import nnAPI
nnapi = nnAPI()



# from flask import jsonify

def get_response_image(image_path):
    pil_img = Image.open(image_path) # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='JPEG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

# server side code




server = flask.Flask(__name__)

# @server.route('/page-2/')
# def redirect():
#     print('here')
#     return redirect(url_for('page-1'))

@server.route('/api/v1.0/get_image')
def get_image():

    filename = 'res.jpg'
    return send_file(filename, mimetype='image/jpg')

@server.route('/api/v1.0/coords', methods=['POST'])
def login():

    lon = request.get_json().get('lon', '')
    lat = request.get_json().get('lat', '')
    print("HERE", lon, lat)
    if lon and lat:
        video_capture = cv2.VideoCapture('http://50.246.145.122/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER')
        success, frame = video_capture.read()
        predicted_img = nnapi.make_prediction(frame)
        cv2.imwrite('res.jpg', predicted_img)
        response =  { 'lon' : lon, 'lat': lat}
        return response
    return "error"

# @server.route('/api/v1.0/test', methods=['GET'])
# def get_tasks():
#     return flask.jsonify({'hello': 'world'})

@server.route('/')
def index():
    return 'Hello Flask app'


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/page-1/',
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ], suppress_callback_exceptions=True
)

# app = dash.Dash(
#
# )





# VALID_USERNAME_PASSWORD_PAIRS = {
#     'alexdiana': '113113113'
# }
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )
