# app/services/inference_service.py
import io
import numpy as np
from PIL import Image
from ..exceptions.input_error import InputError
from .model_service import model_service
import tensorflow as tf

def predict_classification(image_bytes):
    try:
        # Pra-pemrosesan gambar
        pil = Image.open(io.BytesIO(image_bytes))
        arr = model_service.preprocess_pil(pil)
        x = np.expand_dims(arr, axis=0)

        # Embedding gambar
        q_emb = model_service.embedding_model(x, training=False).numpy()

        # Hitung jarak ke prototipe
        dists = np.sum((q_emb[:, None, :] - model_service.prototypes[None, :, :]) ** 2, axis=2)
        logits = -dists
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        # Ambil prediksi terbaik dan top-K
        pred_idx = int(np.argmax(probs))
        pred_name = model_service.class_names[pred_idx]
        topk = min(3, len(model_service.class_names))
        top_idx = np.argsort(-probs)[:topk]
        top = [{"class": model_service.class_names[i], "prob": float(probs[i])} for i in top_idx]

        return {
            'label': pred_name,
            'confidence': float(probs[pred_idx]),
            'suggestion': 'Berdasarkan kemiripan visual.', # Saran generik
            'topk': top
        }

    except Exception as e:
        raise InputError(f'Terjadi kesalahan dalam melakukan prediksi: {e}')