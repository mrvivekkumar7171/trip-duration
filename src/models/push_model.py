from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
import sys

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

model_path = sys.argv[1]
gcs_bucket_name = sys.argv[2]

upload_to_gcs(model_path, gcs_bucket_name, model_path)