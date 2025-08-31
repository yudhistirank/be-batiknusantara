# app/handlers.py
from flask import request, jsonify
from ..services.inference_service import predict_classification
from ..services.store_data import store_data
from ..services.get_history import get_history
from ..utils.format_timestamp import format_timestamp_indonesia
from datetime import datetime

def post_predict_handler():
    file = request.files['image']
    image_bytes = file.read()
    
    result = predict_classification(image_bytes)

    created_at = datetime.utcnow()
    data = {
        "filename": file.filename,
        "result": result['label'],
        "confidence": result['confidence'],
        "suggestion": result['suggestion'],
        "timestamp": created_at
    }

    doc_id = store_data(data)

    return jsonify({
        "id": doc_id,
        "timestamp": format_timestamp_indonesia(created_at),
        "prediction": result['label'],
        "confidence": result['confidence'],
        "suggestion": result['suggestion'],
        "topk": result['topk']
    }), 200

def get_predict_histories_handler():
    ... # Tidak perlu diubah, biarkan seperti semula