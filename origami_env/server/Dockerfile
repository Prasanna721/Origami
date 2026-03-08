FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

COPY server/ ./server/
COPY viewer/ ./viewer/
COPY openenv.yaml ./

USER user

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
