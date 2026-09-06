from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.models import BlobEntry
from refaat.errors import NotFoundError, ValidationError
from refaat.security import safe_join


def _is_missing_s3_key(exc: Exception) -> bool:
    """True when a botocore error means the object is absent, not a real failure.

    A missing key surfaces as a ``ClientError`` carrying ``NoSuchKey``/``NotFound`` (or a 404 on an
    S3-compatible backend), so erasure verification can treat it as absent instead of propagating a
    failure the way ``LocalBlobStore.open`` maps ``FileNotFoundError`` to ``NotFoundError``.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "NotFound", "404"} or status == 404


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def put(self, key: str, content: bytes, *, media_type: str) -> str:
        del media_type
        target = safe_join(self.root, key)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".refaat-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return f"file://{target}"

    def open(self, uri: str) -> BinaryIO:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValidationError("Local blob store only accepts file:// URIs")
        path = Path(parsed.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValidationError("Blob URI escapes the configured root")
        try:
            return path.open("rb")
        except FileNotFoundError as exc:
            raise NotFoundError("Blob not found") from exc

    def delete(self, uri: str) -> None:
        parsed = urlparse(uri)
        path = Path(parsed.path).resolve()
        if parsed.scheme != "file" or not path.is_relative_to(self.root):
            raise ValidationError("Invalid local blob URI")
        path.unlink(missing_ok=True)

    def list_blobs(self) -> Iterator[BlobEntry]:
        for path in self.root.rglob("*"):
            # In-flight writes land on a temp file that os.replace renames onto the
            # final key; skipping it keeps the sweep from racing an active put.
            if not path.is_file() or path.name.startswith(".refaat-"):
                continue
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            yield BlobEntry(uri=f"file://{path}", modified_at=modified_at)

    def modified_at(self, uri: str) -> datetime | None:
        parsed = urlparse(uri)
        path = Path(parsed.path).resolve()
        if parsed.scheme != "file" or not path.is_relative_to(self.root):
            raise ValidationError("Invalid local blob URI")
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            return None


class S3BlobStore:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        region: str,
        access_key_id: str | None,
        secret_access_key: str | None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install refaat[s3] for S3 blob storage") from exc
        # boto3 egresses outside the httpx seam, so this endpoint check is the only in-library egress
        # governance for the blob store; the deploy-layer NetworkPolicy is the backstop for the rest.
        if egress_policy is not None and endpoint_url:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(endpoint_url, EgressPurpose.BLOB))
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def put(self, key: str, content: bytes, *, media_type: str) -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=media_type,
            ServerSideEncryption="AES256",
        )
        return f"s3://{self.bucket}/{key}"

    def open(self, uri: str) -> BinaryIO:
        from botocore.exceptions import ClientError

        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValidationError("Invalid S3 blob URI")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=parsed.path.lstrip("/"))
        except ClientError as exc:
            # Parity with LocalBlobStore.open: an absent object is NotFoundError, so erasure
            # verification sees the blob gone instead of a raw ClientError that would falsely
            # record FAILED and skip the audit event after a correct delete.
            if _is_missing_s3_key(exc):
                raise NotFoundError("Blob not found") from exc
            raise
        return io.BytesIO(response["Body"].read())

    def delete(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValidationError("Invalid S3 blob URI")
        self.client.delete_object(Bucket=self.bucket, Key=parsed.path.lstrip("/"))

    def list_blobs(self) -> Iterator[BlobEntry]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for entry in page.get("Contents", ()):
                yield BlobEntry(uri=f"s3://{self.bucket}/{entry['Key']}", modified_at=entry["LastModified"])

    def modified_at(self, uri: str) -> datetime | None:
        from botocore.exceptions import ClientError

        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValidationError("Invalid S3 blob URI")
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=parsed.path.lstrip("/"))
        except ClientError as exc:
            if _is_missing_s3_key(exc):
                return None
            raise
        last_modified = response.get("LastModified")
        return last_modified if isinstance(last_modified, datetime) else None

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
