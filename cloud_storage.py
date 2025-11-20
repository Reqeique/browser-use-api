import os
import boto3
import logging
import tarfile
import tempfile
from pathlib import Path
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class CloudStorage:
    def __init__(self):
        self.endpoint_url = os.getenv('R2_ENDPOINT_URL')
        self.access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        self.secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        self.bucket_name = os.getenv('R2_BUCKET_NAME')

        if not all([self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket_name]):
            logger.warning("R2 credentials not fully configured. Cloud storage disabled.")
            self.client = None
        else:
            try:
                # Configure boto3 with timeouts
                config = Config(
                    connect_timeout=10,
                    read_timeout=120,  # Increased for large archives
                    retries={'max_attempts': 3, 'mode': 'adaptive'}
                )
                
                self.client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    config=config
                )
                logger.info("Cloud storage client initialized with timeouts.")
            except Exception as e:
                logger.error(f"Failed to initialize cloud storage client: {e}")
                self.client = None

    def upload_directory(self, local_dir: str, s3_prefix: str):
        """Uploads a directory as a compressed tar.gz archive to R2."""
        if not self.client:
            return

        if not os.path.exists(local_dir):
            logger.warning(f"Local directory {local_dir} does not exist. Skipping upload.")
            return

        logger.info(f"Compressing and uploading {local_dir} to {self.bucket_name}/{s3_prefix}.tar.gz")
        
        # Create temporary tar.gz file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Create compressed archive
            logger.info(f"Creating archive from {local_dir}...")
            with tarfile.open(tmp_path, 'w:gz') as tar:
                tar.add(local_dir, arcname='.')
            
            archive_size = os.path.getsize(tmp_path) / (1024 * 1024)  # MB
            logger.info(f"Archive created: {archive_size:.2f} MB")
            
            # Upload single archive file
            s3_key = f"{s3_prefix}.tar.gz"
            logger.info(f"Uploading archive to {s3_key}...")
            self.client.upload_file(tmp_path, self.bucket_name, s3_key)
            
            logger.info(f"Upload completed successfully: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload directory: {e}")
            raise
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def download_directory(self, s3_prefix: str, local_dir: str):
        """Downloads and extracts a compressed tar.gz archive from R2."""
        if not self.client:
            logger.warning("Cloud storage client not initialized. Skipping download.")
            return

        s3_key = f"{s3_prefix}.tar.gz"
        logger.info(f"Downloading archive {self.bucket_name}/{s3_key} to {local_dir}")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Check if archive exists
            try:
                self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.info(f"No archive found in R2 for {s3_prefix}. Profile might be new.")
                    return
                raise
            
            # Download archive
            logger.info(f"Downloading archive from {s3_key}...")
            self.client.download_file(self.bucket_name, s3_key, tmp_path)
            
            archive_size = os.path.getsize(tmp_path) / (1024 * 1024)  # MB
            logger.info(f"Archive downloaded: {archive_size:.2f} MB")
            
            # Ensure target directory exists
            os.makedirs(local_dir, exist_ok=True)
            
            # Extract archive
            logger.info(f"Extracting archive to {local_dir}...")
            with tarfile.open(tmp_path, 'r:gz') as tar:
                tar.extractall(path=local_dir)
            
            logger.info(f"Download and extraction completed successfully")
        except Exception as e:
            logger.error(f"Failed to download directory: {e}")
            raise
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def delete_directory(self, s3_prefix: str):
        """Deletes the profile archive from R2."""
        if not self.client:
            return

        s3_key = f"{s3_prefix}.tar.gz"
        logger.info(f"Deleting {self.bucket_name}/{s3_key}")
        
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deletion completed successfully: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to delete archive: {e}")
            raise