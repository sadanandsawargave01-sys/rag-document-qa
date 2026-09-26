import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET, S3_REGION


s3_client = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


def upload_file_to_s3(file_bytes: bytes, file_uuid: str) -> str:
    """
    Upload file to S3 and return its S3 URL.
    """
    try:
        s3_key = f"uploads/{file_uuid}"
        s3_client.put_object(Bucket=S3_BUCKET,Key=s3_key,  Body=file_bytes,ContentType="application/pdf",ContentDisposition="inline")
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        raise Exception("Error in Uploading file in s3:- {}".format(e))


def get_presigned_url(s3_url: str, expiration: int = 3600, disposition: str = "inline") -> str:
    """
    Generate a presigned URL to share an S3 object

    :param object_key: S3 object key (path to file)
    :param expiration: Time in seconds for the presigned URL to remain valid (default 1 hour)
    :return: Presigned URL as string. If error, returns None.
    """
    try:
        file_uuid = s3_url.rstrip("/").split("/")[-1]
        object_key = f"uploads/{file_uuid}"
        
#        response = s3_client.generate_presigned_url("get_object",
#            Params={"Bucket": S3_BUCKET, "Key": object_key, 'ResponseContentDisposition': disposition},
#            ExpiresIn=expiration
#        )
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": object_key,
                "ResponseContentDisposition": "inline",
                "ResponseContentType": "application/pdf"
            },
            ExpiresIn=expiration
        )
    except ClientError as e:
        print(f"❌ Error generating presigned URL: {e}")
        return None

    return response
