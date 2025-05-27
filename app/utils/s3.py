# from io import BytesIO
import os
import boto3
import uuid
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class S3:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id = settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY,
        )

    def save_s3(self, image_path):
        # buffer = BytesIO()
        # image.save(buffer, format = "PNG")
        # buffer.seek(0)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"{image_path} 해당 경로에 이미지 파일이 존재하지 않습니다.")

        unique_id = str(uuid.uuid4())
        s3_key = f"assets/images/{unique_id}.png"

        with open(image_path, "rb") as f:
            self.s3_client.upload_fileobj(
                f,
                settings.S3_BUCKET_NAME,
                s3_key,
                ExtraArgs={"ContentType": "image/png"}
            )

        # self.s3_client.upload_fileobj(
        #     buffer,
        #     settings.S3_BUCKET_NAME,
        #     s3_key,
        #     ExtraArgs={"ContentType": "image/png"}
        # )

        generated_image_url = f"https://img.onthe-top.com/{unique_id}.png"
        logger.info(f"생성된 이미지 URL = {generated_image_url}")
        
        return generated_image_url