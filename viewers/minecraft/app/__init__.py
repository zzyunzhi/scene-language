from flask import Flask

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
