FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]
