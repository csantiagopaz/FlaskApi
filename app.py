from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'This is my first API call!'

@app.route('/model/<model_name>')
def model(model_name):
    assert model_name == request.view_args['model_name']
    model_dict = {"Model1": Model1}