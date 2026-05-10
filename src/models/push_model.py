import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from logger import infologger

class GCSPush:
    """Create a GCS client (automatically handles credentials if set up correctly)
    """
    def __init__(self):
        try:
            self.client = storage.Client()
        except Exception as e:
            infologger.info(f'Not able to Connect to GCS :  {e}')

    def push(self, local_file_path, gcs_file_path, bucket_name):
        """Get the bucket, create a blob (the object that will store your file in GCS) and upload the file.
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            infologger.info(f"File uploaded successfully to {bucket_name}/{gcs_file_path}\n")
            
        except FileNotFoundError:
            infologger.info(f"The file {local_file_path} was not found.")
        except DefaultCredentialsError:
            infologger.info("Credentials not available. Please check your GCP authentication.")
        except Exception as e:
            infologger.info(f'Not able to upload to GCS :  {e}')

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    model_path = home_dir / sys.argv[1]
    model_path_str = model_path.as_posix()
    model_output_path = sys.argv[2]
    gcs_bucket_name = sys.argv[3]

    try:
        gcs = GCSPush()
        gcs.push(model_path_str, model_output_path, gcs_bucket_name)
    except Exception as e:
        infologger.info(f'Failed in pushing a model to GCS :  {e}')
    else:
        infologger.info('Model pushed to GCS successfully')