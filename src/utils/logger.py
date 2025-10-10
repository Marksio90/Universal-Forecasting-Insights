from loguru import logger
import sys, pathlib, os
def configure_logger():
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL","INFO"))
    pathlib.Path("logs").mkdir(exist_ok=True)
    logger.add("logs/app.log", level="DEBUG", rotation="5 MB", retention="7 days", enqueue=True)
    return logger
