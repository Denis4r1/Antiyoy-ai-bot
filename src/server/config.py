import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent.parent.parent
WEB_DIR = BASE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Настройки приложения
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Настройки игры
GAME_START_COUNTDOWN = int(os.getenv("GAME_START_COUNTDOWN", "5")) # секунд
ROOM_DELETION_DELAY = int(os.getenv("ROOM_DELETION_DELAY", "20")) # секунд
# ^ для того, чтобы при реконнекте (например обновление страницы) комната не удалялась 

# Prometheus (для сбора метрик со всех запущенный процессов)
PROMETHEUS_MULTIPROC_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc")

# CORS (* для дебага)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
