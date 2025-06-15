import logging
import logging.config
import sys
from ..config import LOG_LEVEL, DEBUG


def setup_logging():
    """Настройка логирования для приложения"""

    log_format = (
        "%(asctime)s | %(levelname)8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": "%(asctime)s | ACCESS | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
            "access_console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "access",
            },
        },
        "loggers": {
            "": {  # root logger
                "level": LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "game_server": {
                "level": LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "game_server.lobby": {
                "level": LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "game_server.game": {
                "level": LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "game_server.api": {
                "level": LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO" if DEBUG else "WARNING",
                "handlers": ["access_console"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str):
    """Получить логгер с префиксом game_server"""
    return logging.getLogger(f"game_server.{name}")
