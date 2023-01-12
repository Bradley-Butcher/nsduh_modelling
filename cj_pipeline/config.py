from pathlib import Path
import logging
import logging.config

from rich.logging import RichHandler
import sys

BASE_DIR = Path(__file__).parents[1]
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CRIMES = ['aggravated assault', 'drugs', 'dui', 'property', 'robbery', 'sex offense', 'simple assault']

NEULAW_GROUP = ['def.gender', 'def.race', 'offense_category']  # 'age_cat'
CRIMES_GROUP = ['offender_sex', 'offender_race', 'crime_recode',  'offender_age']
NEULAW_TO_NCVS = NEULAW_GROUP + ['age_ncvs']
NEULAW_TO_NSDUH = NEULAW_GROUP + ['age_nsduh']
# CAVEAT: *order* of CRIMES_GROUP and NEULAW_TO_* + ['age_*'] must match
#  -> see synthetic_assignment.py for usage (left_on/right_on pd.merge)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

print(logging)

logging.config.dictConfig(logging_config)
logging.captureWarnings(True)
logger = logging.getLogger()

logger.handlers[0] = RichHandler(markup=True)