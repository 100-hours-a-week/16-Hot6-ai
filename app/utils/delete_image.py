import os
import glob
import logging

logger = logging.getLogger(__name__)

def delete_images(folder_path = None):
    if folder_path is None:
        folder_path = "/temp/"

    if not os.path.exists(folder_path):
        logger.info(f"{folder_path} is not exists")
        return
    
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    for file_path in png_files:
        try:
            os.remove(file_path)
            logger.info("Image deleted Successfully")
        except Exception as e:
            logger.error(f"delete_images is failed: {file_path} → {e}")