import sys
import os
from dotenv import load_dotenv

# Menambahkan direktori induk ke jalur impor Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.init import create_app

load_dotenv()
app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
