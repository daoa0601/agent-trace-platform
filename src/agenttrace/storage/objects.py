"""Small object-store abstraction with immutable-write semantics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


def validate_key(key: str) -> str:
    """Reject absolute and traversing object keys."""
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe object key: {key!r}")
    return path.as_posix()


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    created: bool
    etag: str | None = None


class ObjectStore(Protocol):
    """Minimal API shared by local files and S3-compatible storage."""

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        if_absent: bool = True,
    ) -> ObjectInfo: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str = "") -> Iterator[str]: ...


class FileObjectStore:
    """Filesystem object store for development and deterministic tests."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = validate_key(key)
        path = (self.root / safe_key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"object key escapes storage root: {key!r}")
        return path

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        if_absent: bool = True,
    ) -> ObjectInfo:
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if if_absent:
            try:
                with path.open("xb") as handle:
                    handle.write(data)
                return ObjectInfo(key=validate_key(key), size=len(data), created=True)
            except FileExistsError:
                return ObjectInfo(key=validate_key(key), size=path.stat().st_size, created=False)
        path.write_bytes(data)
        return ObjectInfo(key=validate_key(key), size=len(data), created=True)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        safe_prefix = validate_key(prefix) if prefix else ""
        search_root = self._path(safe_prefix) if safe_prefix else self.root
        if search_root.is_file():
            yield search_root.relative_to(self.root).as_posix()
            return
        if not search_root.exists():
            return
        for path in sorted(search_root.rglob("*")):
            if path.is_file():
                yield path.relative_to(self.root).as_posix()


class S3ObjectStore:
    """S3-compatible object store using conditional immutable writes."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        import boto3

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        if_absent: bool = True,
    ) -> ObjectInfo:
        from botocore.exceptions import ClientError

        safe_key = validate_key(key)
        kwargs: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": safe_key,
            "Body": data,
            "ContentType": content_type,
        }
        if if_absent:
            kwargs["IfNoneMatch"] = "*"
        try:
            response = self._client.put_object(**kwargs)
            return ObjectInfo(
                key=safe_key,
                size=len(data),
                created=True,
                etag=str(response.get("ETag", "")).strip('"') or None,
            )
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if if_absent and status in {409, 412}:
                head = self._client.head_object(Bucket=self.bucket, Key=safe_key)
                return ObjectInfo(
                    key=safe_key,
                    size=int(head["ContentLength"]),
                    created=False,
                    etag=str(head.get("ETag", "")).strip('"') or None,
                )
            raise

    def get_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=validate_key(key))
        return bytes(response["Body"].read())

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=validate_key(key))
            return True
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                return False
            raise

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        safe_prefix = validate_key(prefix) if prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=safe_prefix):
            for item in page.get("Contents", []):
                yield str(item["Key"])
