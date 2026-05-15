import boto3
from botocore.exceptions import ClientError
from app.config import settings

s3_client = boto3.client("s3")


def upload_to_s3(file_path: str, filename: str) -> str:
    """
    Upload a PDF to S3. Returns the S3 key.
    """
    s3_key = f"documents/{filename}"
    s3_client.upload_file(file_path, settings.S3_BUCKET, s3_key)
    return s3_key


def file_exists_in_s3(filename: str) -> bool:
    """
    Check if a file already exists in S3.
    Returns True if it does, False if not.
    """
    s3_key = f"documents/{filename}"
    try:
        s3_client.head_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False