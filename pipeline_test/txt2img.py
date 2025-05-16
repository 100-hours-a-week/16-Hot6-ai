import uuid
import boto3
from PIL import Image
import io
import os
from dotenv import load_dotenv

class TextToImage:
    def __init__(self, prompt):
        load_dotenv()
        self.prompt = prompt
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_prefix = os.getenv("S3_PREFIX")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    def generate_image(self):
        # 임시 샘플 이미지
        image = Image.open("success.jpg").convert("RGB")

        # UUID 기반 파일명 구성
        filename = f"{uuid.uuid4()}.jpg"
        s3_key = f"{self.s3_prefix}/{filename}"

        # 이미지 메모리 버퍼 저장
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # boto3 클라이언트 생성
        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

        # S3 업로드
        s3.upload_fileobj(buffer, self.s3_bucket, s3_key, ExtraArgs={"ContentType": "image/jpeg"})

        # S3 URL 반환
        image_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
        return image_url

        