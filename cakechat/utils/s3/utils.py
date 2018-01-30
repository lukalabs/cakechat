import boto3
from botocore import UNSIGNED
from botocore.client import Config


def get_s3_resource():
    return boto3.resource('s3', config=Config(signature_version=UNSIGNED))
