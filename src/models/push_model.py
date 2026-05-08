from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import shutil

def upload_to_gcs(local_file_path, bucket_name, gcs_file_path):
    try:
        # Create a GCS client (automatically handles credentials if set up correctly)
        client = storage.Client()
        
        # Get the bucket
        bucket = client.bucket(bucket_name)
        
        # Create a blob (the object that will store your file in GCS)
        blob = bucket.blob(gcs_file_path)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        
        # print(f"File uploaded successfully to {bucket_name}/{gcs_file_path}\n")
        
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found.")
    except DefaultCredentialsError:
        print("Credentials not available. Please check your GCP authentication.")
    except Exception as e:
        print(f"An error occurred: {e}")

local_model_path = 'models/model.joblib'
gcs_bucket_name = 'trip-duration-dvc-storages'
gcs_file_path = 'models/model.joblib'

upload_to_gcs(local_model_path, gcs_bucket_name, gcs_file_path)
shutil.copy(local_model_path, 'model.joblib')