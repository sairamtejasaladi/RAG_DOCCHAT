from loguru import logger

logger.add(
    "docchat_app.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
