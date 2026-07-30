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


def temp_pdf(file_bytes: bytes):
    return temp_file(file_bytes, ".pdf")
