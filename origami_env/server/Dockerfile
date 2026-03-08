FROM python:3.11-slim

WORKDIR /app

# Python dependencies (lightweight — no matplotlib/Pillow/imageio)
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy server code
COPY server/ ./server/

# Copy Three.js viewer (static HTML/JS)
COPY viewer/ ./viewer/

WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
