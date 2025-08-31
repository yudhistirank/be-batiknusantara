# Gunakan image Python yang ringan
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project ke container
COPY . .

# Expose port
EXPOSE 8080

# Jalankan aplikasi
CMD ["gunicorn", "-b", "0.0.0.0:8080", "run:app"]
