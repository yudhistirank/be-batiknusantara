import os
from dotenv import load_dotenv

# Memuat environment variables dari file .env
load_dotenv()

# BENAR: Mengimpor create_app langsung dari paket 'app'.
# Python akan otomatis mencari di app/__init__.py
from app import create_app

# Membuat instance aplikasi
app = create_app()

if __name__ == "__main__":
    # Mengambil port dari environment atau default ke 8080
    port = int(os.getenv("PORT", 8080))
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1"]
    
    print(f"[INFO] Starting Flask development server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)