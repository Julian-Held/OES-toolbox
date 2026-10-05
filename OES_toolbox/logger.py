import logging

LOGGER_NAME = "OESToolbox"

def configure_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if configuration is called more than once.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | "
                "%(name)s.%(class_name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                defaults={"class_name": ""},
            )
        )
        logger.addHandler(handler)


class ContextLogger(logging.LoggerAdapter):
    """A logger adapter that adds contextual information, such as the class name, to log records."""

    def __init__(self, instance: object|None, level="info", context: None|dict =None):
        # TODO: 'level' kwarg which is no longer used as it sets the level of the logger globally.
        # Instead, level should be set on the handler(s)
        extra = dict(context or {})
        if instance is not None:
            if isinstance(instance, type):
                extra.setdefault("class_name", instance.__name__)
            else:
                extra.setdefault("class_name", type(instance).__name__)
        super().__init__(logging.getLogger(LOGGER_NAME), extra)
