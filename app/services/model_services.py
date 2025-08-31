# app/services/model_service.py
import os
import io
import tempfile
import zipfile
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model
from google.cloud import storage

# -------- Config --------
EMBED_DIM = 128
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16

# Inisialisasi Google Cloud Storage Client
storage_client = storage.Client()


def get_embedding_model(input_shape=(224, 224, 3), emb_dim=128):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    out = layers.Dense(emb_dim, activation=None)(x)
    return Model(base.input, out, name="VGG16_Embedding")


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    w, h = pil_img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = pil_img.crop((left, top, left + m, top + m))
    img = img.convert("RGB").resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def list_classes(base_dir):
    return [
        d
        for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ]


class ModelService:
    def __init__(self):
        self.embedding_model = None
        self.class_names = None
        self.prototypes = None

    def load_and_build(self):
        # Ambil URL dari variabel lingkungan
        model_url = os.environ.get("MODEL_URL")
        test_data_url = os.environ.get("TEST_DATA_URL")

        if not model_url or not test_data_url:
            raise ValueError("Variabel lingkungan MODEL_URL atau TEST_DATA_URL tidak ditemukan.")

        # 1. Unduh bobot model dari GCS
        model_weights_path = "model_weights.h5"
        print(f"Mengunduh model dari {model_url}")
        try:
            blob = storage_client.get_bucket(model_url.split('/')[2]).blob('/'.join(model_url.split('/')[3:]))
            blob.download_to_filename(model_weights_path)
            print("Bobot model berhasil diunduh.")
        except Exception as e:
            raise RuntimeError(f"Gagal mengunduh model: {e}")

        # 2. Bangun arsitektur model dan muat bobot
        self.embedding_model = get_embedding_model(emb_dim=EMBED_DIM)
        self.embedding_model.load_weights(model_weights_path)
        print("Model berhasil dimuat.")

        # 3. Unduh dan ekstrak data untuk prototipe
        temp_dir = tempfile.mkdtemp()
        test_zip_path = os.path.join(temp_dir, "test_data.zip")
        print(f"Mengunduh data uji dari {test_data_url}")
        try:
            blob = storage_client.get_bucket(test_data_url.split('/')[2]).blob('/'.join(test_data_url.split('/')[3:]))
            blob.download_to_filename(test_zip_path)
            print("Data uji berhasil diunduh.")

            # Ekstrak file zip
            with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print("Data uji berhasil diekstrak.")

        except Exception as e:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"Gagal mengunduh data uji: {e}")
        
        test_dir = os.path.join(temp_dir, os.listdir(temp_dir)[1])

        # 4. Bangun prototipe
        print("Membangun prototipe...")
        all_protos = []
        self.class_names = list_classes(test_dir)
        for cls in self.class_names:
            cdir = os.path.join(test_dir, cls)
            if not os.path.isdir(cdir): continue
            
            files = [f for f in os.listdir(cdir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not files:
                print(f"Kelas '{cls}' tidak punya gambar.")
                continue

            batch_embs = []
            for i in range(0, len(files), BATCH_SIZE):
                batch_files = files[i:i+BATCH_SIZE]
                batch = []
                for fname in batch_files:
                    arr = preprocess_pil(Image.open(os.path.join(cdir, fname)))
                    batch.append(arr)
                batch = np.array(batch, dtype=np.float32)
                embs = self.embedding_model(batch, training=False).numpy()
                batch_embs.append(embs)
            embs = np.concatenate(batch_embs, axis=0)
            proto = np.mean(embs, axis=0)
            all_protos.append(proto)

        if not all_protos:
            shutil.rmtree(temp_dir)
            raise ValueError("Tidak ada prototipe yang berhasil dibuat.")
            
        self.prototypes = np.stack(all_protos, axis=0)
        shutil.rmtree(temp_dir)
        
        print(f"[INFO] Kelas ditemukan: {self.class_names}")
        print(f"Ukuran prototipe: {self.prototypes.shape}")


# Instansiasi service ini saat startup
model_service = ModelService()
model_service.load_and_build()