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
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()
MODEL_URL = os.getenv("MODEL_URL")
TEST_DATA_URL = os.getenv("TEST_DATA_URL")

EMBED_DIM = 128
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16

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

def download_file(url, suffix=""):
    response = requests.get(url)
    response.raise_for_status()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(response.content)
    temp.close()
    return temp.name

class ModelService:
    def __init__(self):
        self.embedding_model = None
        self.class_names = None
        self.prototypes = None

    def load_and_build(self):
        model_url = os.environ.get("MODEL_URL")
        test_data_url = os.environ.get("TEST_DATA_URL")

        if not model_url or not test_data_url:
            raise ValueError("Variabel lingkungan MODEL_URL atau TEST_DATA_URL tidak ditemukan.")

        # 1. Load model weights (.h5) dari URL atau lokal
        if urlparse(model_url).scheme in ("http", "https"):
            print(f"Mengunduh model dari {model_url}")
            model_weights_path = download_file(model_url, suffix=".h5")
        else:
            print(f"Menggunakan model lokal dari {model_url}")
            if not os.path.exists(model_url):
                raise FileNotFoundError(f"Model file not found at {model_url}")
            model_weights_path = model_url

        # 2. Build model dan load weights
        self.embedding_model = get_embedding_model(emb_dim=EMBED_DIM)
        self.embedding_model.load_weights(model_weights_path)
        print("Model berhasil dimuat.")

        # 3. Load test data (zip) dari URL atau lokal
        temp_dir = tempfile.mkdtemp()
        if urlparse(test_data_url).scheme in ("http", "https"):
            print(f"Mengunduh data uji dari {test_data_url}")
            test_zip_path = download_file(test_data_url, suffix=".zip")
        else:
            print(f"Menggunakan data uji lokal dari {test_data_url}")
            if not os.path.exists(test_data_url):
                shutil.rmtree(temp_dir)
                raise FileNotFoundError(f"Test data file not found at {test_data_url}")
            test_zip_path = test_data_url

        # Ekstrak file zip
        try:
            with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print("Data uji berhasil diekstrak.")
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"Gagal mengekstrak data uji: {e}")

        # Cari folder data uji
        test_dir_candidates = [
            os.path.join(temp_dir, d)
            for d in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, d))
        ]
        if not test_dir_candidates:
            shutil.rmtree(temp_dir)
            raise ValueError("Folder data uji tidak ditemukan setelah ekstraksi.")
        test_dir = test_dir_candidates[0]

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