import os

import boto3


def get_s3_client():
    # modify to your connection approach if not using access/secret keys
    s3_client = boto3.client(
        "s3",
    )
    return s3_client


def split_s3_file(s3_filename):
    s3_filename_split = s3_filename.replace("s3://", "").split("/")
    bucket = s3_filename_split[0]
    key = "/".join(s3_filename_split[1:])
    return bucket, key


def download_file_from_s3(s3_filename, local_fname):
    bucket, key = split_s3_file(s3_filename)
    return get_s3_client().download_file(bucket, key, local_fname)


def upload_file_to_s3(s3_filename, local_fname):
    bucket, key = split_s3_file(s3_filename)
    return get_s3_client().upload_file(local_fname, bucket, key)
