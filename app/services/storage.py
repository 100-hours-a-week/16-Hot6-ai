# fastapi_project/app/services/storage.py
import io, boto3, httpx, uuid
from ..core.config import get_settings

settings = get_settings()
s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.S3_REGION,
)

async def download_image(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        return r.content

async def upload_image(image_bytes: bytes, suffix: str = ".png") -> str:
    key = f"generated/{uuid.uuid4()}{suffix}"
    s3.upload_fileobj(io.BytesIO(image_bytes), settings.S3_BUCKET, key)
    return f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/{key}"

async def post_callback(initial_url: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            settings.CALL_BACK_URL,
            json={"initial_image_url": initial_url, "classify": True},
            timeout=10,
        )
