import os

from cakechat.utils.files_utils import ensure_dir
from cakechat.utils.logger import get_logger


class S3Bucket(object):
    def __init__(self, bucket_client):
        self._logger = get_logger(__name__)
        self._bucket_client = bucket_client

    def download(self, remote_file_name, local_file_name):
        """
        Download file from AWS S3 to the local one
        """
        remote_file_name = os.path.normpath(remote_file_name)

        # create dir if not exists for storing file from s3
        ensure_dir(os.path.dirname(local_file_name))

        self._logger.info('Getting file %s from AWS S3 and saving it as %s' % (remote_file_name, local_file_name))
        self._bucket_client.download_file(remote_file_name, local_file_name)
        self._logger.info('Got file %s from S3' % remote_file_name)

    def upload(self, local_file_name, remote_file_name):
        """
        Upload local file to AWS S3 bucket
        """
        self._logger.info('Saving file {} in amazon S3 as {}'.format(local_file_name, remote_file_name))
        self._bucket_client.upload_file(local_file_name, remote_file_name)
        self._logger.info('File %s saved to %s on S3' % (local_file_name, remote_file_name))
