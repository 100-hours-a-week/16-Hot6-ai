import os
import glob
import logging

logger = logging.getLogger(__name__)

def delete_images():
    folder_path = "/temp/"

    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    for file_path in png_files:
        try:
            os.remove(file_path)
            logger.info("이미지 삭제 완료")
        except Exception as e:
            logger.error(f"이미지 삭제 실패: {file_path} → {e}")