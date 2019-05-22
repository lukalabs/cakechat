import os
from functools import partial

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from cakechat.utils.files_utils import AbstractFileResolver, PackageResolver, extract_tar
from cakechat.utils.logger import WithLogger
from cakechat.utils.s3 import S3Bucket


class S3FileResolver(AbstractFileResolver, WithLogger):
    """
    Tries to download file from AWS S3 if it does not exist locally
    """

    def __init__(self, file_path, bucket_name, remote_dir):
        super(S3FileResolver, self).__init__(file_path)
        WithLogger.__init__(self)

        self._bucket_name = bucket_name
        self._remote_dir = remote_dir

    @staticmethod
    def init_resolver(**kwargs):
        """
        Method helping to set once some parameters like bucket_name and remote_dir
        :param kwargs:
        :return: partially initialized class object
        """
        return partial(S3FileResolver, **kwargs)

    def _get_remote_path(self):
        return '%s/%s' % (self._remote_dir, os.path.basename(self._file_path))

    def _resolve(self):
        remote_path = self._get_remote_path()

        try:
            bucket = S3Bucket(get_s3_resource().Bucket(self._bucket_name))
            bucket.download(remote_path, self._file_path)
            return True
        except Exception as e:
            self._logger.warn('File can not be downloaded from AWS S3 because: %s' % str(e))

        return False


def get_s3_resource():
    return boto3.resource('s3', config=Config(signature_version=UNSIGNED))


def get_s3_model_resolver(bucket_name, remote_dir):
    return PackageResolver.init_resolver(
        package_file_resolver_factory=S3FileResolver.init_resolver(bucket_name=bucket_name, remote_dir=remote_dir),
        package_file_ext='tar.gz',
        package_extractor=extract_tar)
