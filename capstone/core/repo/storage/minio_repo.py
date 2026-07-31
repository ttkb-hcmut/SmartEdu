import io
import re
import logging
import hashlib
from datetime import timedelta
from typing import List, Optional

from minio import Minio
from minio.error import S3Error
from core.config import Minio_conf

logger = logging.getLogger(__name__)


def clean_topic_slug(text: Optional[str]) -> str:
    # tidy a heading into a folder-safe slug
    if not text:
        return ""
    text = re.sub(r"\.[a-z0-9]+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text).lower().strip()
    text = re.sub(r"\s+", "_", text)
    return text[:40]


def make_topic_name(file_name: str, heading: Optional[str], chunk_id: str) -> str:
    # unique topic per chunk: readable slug + short hash so no two collide
    base = clean_topic_slug(heading) or clean_topic_slug(file_name) or "topic"
    h = hashlib.sha1(f"{file_name}/{chunk_id}".encode("utf-8")).hexdigest()[:6]
    return f"{base}_{h}"


def validate_file_names(file_names: List[str]) -> None:
    seen = set()
    for name in file_names:
        if not name or not name.strip():
            raise ValueError("file name is blank")
        if "/" in name or "\\" in name or any(ord(char) < 32 for char in name):
            raise ValueError(f"file name is unsafe: {name}")
        if name in seen:
            raise ValueError(f"file name is duplicated: {name}")
        seen.add(name)


class MinioDB:
    def __init__(self, config: Minio_conf = Minio_conf()):
        self.client = Minio(
            endpoint=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure
        )
        self.bucket_name = "courses"
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")

    # ─────────────────────────────────────────────
    # raw upload staging (presigned PUT lands here)
    #   courses/{course}/_raw/{filename}
    # ─────────────────────────────────────────────
    def raw_object_name(self, course_name: str, file_name: str) -> str:
        # staging path for an uploaded, not-yet-ingested pdf
        return f"{course_name}/_raw/{file_name}"

    def presigned_put_url(self, course_name: str, file_name: str, expiry_minutes: int = 30) -> str:
        # short-lived url; browser PUTs the pdf straight to minio, no proxy
        object_name = self.raw_object_name(course_name, file_name)
        return self.client.presigned_put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            expires=timedelta(minutes=expiry_minutes),
        )

    def get_object_bytes(self, object_name: str) -> bytes:
        # read a whole object into memory
        response = self.client.get_object(self.bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def download_object(self, object_name: str, file_path: str) -> None:
        self.client.fget_object(self.bucket_name, object_name, file_path)

    def object_revision(self, object_name: str) -> str:
        stat = self.client.stat_object(self.bucket_name, object_name)
        revision = getattr(stat, "version_id", None) or getattr(stat, "etag", None)
        if not revision:
            raise ValueError(f"object has no cacheable revision: {object_name}")
        return revision

    def object_exists(self, object_name: str) -> bool:
        # true if the object is present
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False

    # ─────────────────────────────────────────────
    # legacy: whole-slide pdf (kept for old test script)
    # ─────────────────────────────────────────────
    def upload_slide(self, name: str, course_name: str, file_data: bytes) -> Optional[str]:
        name = name.split('.')[0]
        try:
            object_name = f"{course_name}/{name}/{name}.pdf"
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type="application/pdf"
            )
            return f"minio://{self.bucket_name}/{object_name}"
        except S3Error as e:
            logger.error(f"Upload slide error: {e}")
            return None

    # ─────────────────────────────────────────────
    # per-topic output (one tiny page-range pdf per chunk)
    #   courses/{course}/{topic}/page.pdf
    #   courses/{course}/{topic}/chunks/{chunk_id}.txt
    # ─────────────────────────────────────────────
    def upload_topic_pdf(self, topic: str, course_name: str, file_data: bytes) -> Optional[str]:
        # store the small page pdf the frontend will show
        try:
            object_name = f"{course_name}/{topic}/page.pdf"
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type="application/pdf",
            )
            return f"minio://{self.bucket_name}/{object_name}"
        except S3Error as e:
            logger.error(f"Upload topic pdf error: {e}")
            return None

    def upload_chunk(self, chunk_id: str, content: str, topic: str, course_name: str) -> Optional[str]:
        # store chunk text next to its page pdf
        try:
            content_bytes = content.encode("utf-8")
            object_name = f"{course_name}/{topic}/chunks/{chunk_id}.txt"
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(content_bytes),
                length=len(content_bytes),
                content_type="text/plain",
            )
            return f"minio://{self.bucket_name}/{object_name}"
        except S3Error as e:
            logger.error(f"Upload chunk error: {e}")
            return None

    # ─────────────────────────────────────────────
    # read / list (frontend discovery + viewer)
    # ─────────────────────────────────────────────
    def list_courses(self) -> List[str]:
        # top-level folders, one per course
        seen = set()
        for obj in self.client.list_objects(self.bucket_name, recursive=False):
            name = (obj.object_name or "").rstrip("/")
            if name:
                seen.add(name.split("/")[0])
        return sorted(seen)

    def list_topics(self, course_name: str) -> List[str]:
        # topic folders under a course, skipping the _raw staging area
        seen = set()
        prefix = f"{course_name}/"
        for obj in self.client.list_objects(self.bucket_name, prefix=prefix, recursive=False):
            name = (obj.object_name or "").rstrip("/")
            tail = name[len(prefix):]
            top = tail.split("/")[0]
            if top and top != "_raw":
                seen.add(top)
        return sorted(seen)

    def topic_pdf_object(self, course_name: str, topic: str) -> str:
        # object key of a topic's page pdf
        return f"{course_name}/{topic}/page.pdf"
