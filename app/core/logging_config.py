import logging
from logging.config import dictConfig

class SuppressHealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            "GET /health" in record.getMessage() or
            "GET /ping" in record.getMessage()
        )

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "suppress_healthcheck": {
            "()": SuppressHealthCheckFilter
        }
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["suppress_healthcheck"]
        }
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO"
    }
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)
