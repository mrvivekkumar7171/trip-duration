"""
    A good python file must have functions, logging, docstrings, type hints and error handling.

    Order of logging levels: DEBUG < INFO < WARNING < ERROR < CRITICAL

    - DEBUG : Used in development
    - INFO : Used to save important information like file saved, model trained, etc.
    - WARNING : Something unexpected happened, but the program is still running. Like a deprecated function, or a missing file that is not critical.
    - ERROR : failed to do something, but the program is still running. Like a missing file that is critical, or a failed API call.
    - CRITICAL : Something went wrong, and the program is not able to continue. Like a failed database connection, or a failed model training.
"""
from datetime import datetime
import logging, os

# Custom Formatter for Console
class ColorFormatter(logging.Formatter):
    
    # Define ANSI color codes
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"

    # Map log levels only to the color variable, not the whole string
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):

        RESET = "\x1b[0m"

        # Fetch the right color for the log level
        color = self.COLORS.get(record.levelno, RESET)
        
        # Inject the color strictly around the %(levelname)s
        log_fmt = f"%(asctime)s:{color}%(levelname)s{RESET}:%(module)s:%(message)s"
        
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# logger
logger = logging.getLogger(__name__) # tell file name in logs
logger.setLevel(logging.DEBUG)

# formatter
plain_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(module)s:%(message)s')

log_path= os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
log_file_name= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path= os.path.join(log_path, log_file_name)

# file handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(plain_formatter)
logger.addHandler(file_handler)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(ColorFormatter())
logger.addHandler(console_handler)

# logging.basicConfig(filename=log_file_path, format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s", level=logging.INFO)

if __name__=="__main__":
    logger.debug("Testing debug log")
    logger.info("Testing info log")
    logger.warning("Testing warning log")
    logger.error("Testing error log")
    logger.critical("Testing critical log")