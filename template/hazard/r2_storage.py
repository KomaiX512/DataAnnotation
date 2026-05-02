from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from template.protocol import R2AccessCredentials


def load_r2_credentials_from_env() -> R2AccessCredentials:
    required = {
        "R2_ACCOUNT_ID": os.getenv("R2_ACCOUNT_ID", "").strip(),
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "").strip(),
        "R2_S3_ENDPOINT": os.getenv("R2_S3_ENDPOINT", "").strip(),
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required R2 env vars for real-model path: {', '.join(missing)}"
        )
    return R2AccessCredentials(
        account_id=required["R2_ACCOUNT_ID"],
        bucket_name=required["R2_BUCKET_NAME"],
        s3_endpoint=required["R2_S3_ENDPOINT"],
        access_key_id=required["R2_ACCESS_KEY_ID"],
        secret_access_key=required["R2_SECRET_ACCESS_KEY"],
        token=os.getenv("R2_TOKEN"),
        public_bucket_url=os.getenv("R2_PUBLIC_BUCKET_URL"),
    )


def upload_checkpoint_to_r2(
    local_checkpoint: Path,
    *,
    object_key: str,
    creds: R2AccessCredentials,
) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for Cloudflare R2 uploads.") from exc
    if not local_checkpoint.exists():
        raise FileNotFoundError(f"Local checkpoint missing for R2 upload: {local_checkpoint}")

    client = boto3.client(
        "s3",
        endpoint_url=creds.s3_endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
    )
    client.upload_file(str(local_checkpoint), creds.bucket_name, object_key)
    return f"r2://{creds.bucket_name}/{object_key}"


def delete_checkpoint_prefix_from_r2(
    *,
    creds: R2AccessCredentials,
    prefix: str,
) -> int:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for Cloudflare R2 cleanup.") from exc
    client = boto3.client(
        "s3",
        endpoint_url=creds.s3_endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
    )
    deleted = 0
    continuation_token = None
    while True:
        list_kwargs = {"Bucket": creds.bucket_name, "Prefix": prefix}
        if continuation_token is not None:
            list_kwargs["ContinuationToken"] = continuation_token
        result = client.list_objects_v2(**list_kwargs)
        objects = [{"Key": item["Key"]} for item in result.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=creds.bucket_name, Delete={"Objects": objects})
            deleted += len(objects)
        if not result.get("IsTruncated"):
            return deleted
        continuation_token = result.get("NextContinuationToken")


def download_checkpoint_from_r2(uri: str, *, creds: R2AccessCredentials, target_path: Path) -> Path:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for Cloudflare R2 downloads.") from exc
    parsed = urlparse(uri)
    if parsed.scheme != "r2":
        raise ValueError(f"Expected r2:// URI, got {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid r2 URI: {uri}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=creds.s3_endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
    )
    client.download_file(bucket, key, str(target_path))
    return target_path
