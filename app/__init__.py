import os
from flask import Flask
from .routes.routes import bp

def create_app():
    # Ini adalah fungsi yang membuat dan mengembalikan instance Flask
    app = Flask(__name__)

    # Tambahkan variabel lingkungan Flask di sini
    # Contoh:
    # app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

    # Daftarkan blueprint routes
    app.register_blueprint(bp)

    return app
