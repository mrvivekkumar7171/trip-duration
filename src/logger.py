from datetime import datetime
import logging, os

infologger = logging.getLogger(__name__)
infologger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

log_path= os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
Log_file= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path= os.path.join(log_path, Log_file)
file_handler = logging.FileHandler(log_file_path)

file_handler.setFormatter(formatter)
infologger.addHandler(file_handler)

# logging.basicConfig(filename=log_file_path, format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s", level=logging.INFO)

if __name__=="__main__":
    infologger.info("Testing log")