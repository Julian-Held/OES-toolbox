import logging


class Logger(logging.LoggerAdapter):
    def __init__(self, instance: object|None, level="info", context: dict = {}):
        self.logger = logging.getLogger("OESToolbox")
        self.logger.setLevel(getattr(logging, level.upper()))
        if len(self.logger.handlers) < 1:
            log_handler = logging.StreamHandler()
            log_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s.%(class)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                defaults={"class": ""},
            )
            log_handler.setFormatter(log_formatter)
            self.logger.addHandler(log_handler)
        if instance is not None:
            context["class"] = instance.__class__.__name__ if not isinstance(instance,type) else instance.__name__
        self.context = context
        super().__init__(self.logger, context)

    def process(self, msg, kwargs):
        # Add context to the log message
        msg, kwargs = super().process(msg, kwargs)
        # return f"{self.context}: {msg}", kwargs
        return msg, kwargs
