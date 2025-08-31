# Menyediakan endpoint /predict untuk menerima gambar dan memanggil layanan prediksi
from flask import Blueprint, request, jsonify
from ..handlers.handlers import post_predict_handler, get_predict_histories_handler

bp = Blueprint('main', __name__)

@bp.route("/", methods=["GET"])
def home():
    return "Batik MaduraKu API sudah ready!"

@bp.route("/predict", methods=["POST"])
def predict():
    if not request.files.get('image'):
        return jsonify({"error": "No image provided"}), 400
    return post_predict_handler()

@bp.route("/predicts", methods=["GET"])
def histories():
    return get_predict_histories_handler()
