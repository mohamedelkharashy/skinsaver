from flask import Flask
from .api import main

def create_app():
    app = Flask(_name_)
    app.register_blueprint(main)
    return app