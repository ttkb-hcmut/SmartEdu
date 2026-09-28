import os
import tempfile
from contextlib import contextmanager


def fetch_raw(minio_repo, course_name: str, file_name: str) -> bytes:
    ## staging object key extension-agnostic
    raw_obj = minio_repo.raw_object_name(course_name, file_name)
    return minio_repo.get_object_bytes(raw_obj)


fetch_raw_pdf = fetch_raw


@contextmanager
def temp_file(file_bytes: bytes, suffix: str):
    ## bytes stay off Prefect state
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes)
    tmp.close()
    try:
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@contextmanager
def temp_raw(minio_repo, course_name: str, file_name: str):
    suffix = os.path.splitext(file_name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    try:
        raw_obj = minio_repo.raw_object_name(course_name, file_name)
        minio_repo.download_object(raw_obj, tmp.name)
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def temp_pdf(file_bytes: bytes):
    return temp_file(file_bytes, ".pdf")
