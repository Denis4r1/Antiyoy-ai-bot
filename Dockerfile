FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn uvicorn[standard]

COPY . .

EXPOSE 8000

ENTRYPOINT ["sh", "-c", "gunicorn src.server.game_interface:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --workers ${GUNICORN_WORKERS}"]
