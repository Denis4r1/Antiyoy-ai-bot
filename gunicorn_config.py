import os
import multiprocessing

# Основные настройки
bind = "0.0.0.0:8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100

# Таймауты
timeout = 30
keepalive = 2
graceful_timeout = 30

# Логирование
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'


# Prometheus multiprocess
def when_ready(server):
    """Called just after the server is started."""
    # директория для метрик
    import os

    metrics_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc")
    os.makedirs(metrics_dir, exist_ok=True)


def worker_exit(server, worker):
    """Called just after a worker has been exited."""

    # Очищаем метрики воркера
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(worker.pid)


# скомпилировать tcs
# uvicorn src.server.app:app --reload --host 0.0.0.0 --port 8081
# или
# gunicorn -c gunicorn_config.py src.server.app:app
#
# А лучше docker-compose -f docker-compose.gameserver.yml --env-file default.env up --build
#
