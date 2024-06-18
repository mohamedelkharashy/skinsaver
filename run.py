from app import create_app
from waitress import serve

app = create_app()

if _name_ == '_main_':
    serve(app, host='0.0.0.0', port=8000)