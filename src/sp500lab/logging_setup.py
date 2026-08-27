"""Console + rotating file logging. Call configure_logging() once at entrypoint."""

from __future__ import annotations

import logging
import logging.handlers

from .paths import LOGS_DIR

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                                           datefmt="%H:%M:%S"))
    root.addHandler(console)

    fileh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "sp500lab.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    fileh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root.addHandler(fileh)

    # yfinance/urllib3 are chatty at INFO
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    _CONFIGURED = True
