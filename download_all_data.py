from google.cloud import storage
import os

def download_public_file(bucket_name, directory_name):
    """Downloads a public blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=directory_name)
    #blobs = bucket.list_blobs()
    for blob in blobs:
        blob = bucket.blob(blob.name)
        blob.download_to_filename(blob.name)

        print(
            "Downloaded public blob {} from bucket {} to {}.".format(
                blob.name, bucket.name, blob.name
            )
        )

os.makedirs('data', exist_ok=True)
os.makedirs('data/newlama', exist_ok=True)
os.makedirs('data/split', exist_ok=True)
os.makedirs('data/updatedlama', exist_ok=True)
download_public_file('continual_learning', 'data/')