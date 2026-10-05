from .config import cfg
from .logging_conf import setup_logging


def main() -> None:
    setup_logging()


__all__ = ["cfg", "main"]
