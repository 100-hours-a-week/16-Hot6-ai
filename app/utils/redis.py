from redis.sentinel import Sentinel
from core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

class RedisSentinel:
    def __init__(self):
        redis_password = settings.REDIS_PASSWORD

        sentinel = Sentinel(
            [(host, int(port)) for host, port in
            (node.strip().split(":") for node in settings.REDIS_SENTINEL_NODES.split(","))],
            socket_timeout=0.5,
            password=redis_password,
            sentinel_kwargs={"password": redis_password}
        )

        self.redis = sentinel.master_for(
            settings.REDIS_MASTER_NAME,
            socket_timeout=0.5,
            password=redis_password,
            decode_responses=True
        )

    def push_completed_image(self, image_url: str, generated_image_url: str = None, products: list[dict] = None) -> None:
        payload = {
            "initial_image_url": image_url,
            "processed_image_url": generated_image_url,
            "products": products
        }
        json_data = json.dumps(payload)
        self.redis.rpush("completed_images", json_data)

    def pop_original_image(self):
        data = self.redis.blpop("original_images", timeout=0)
        if data:
            _, json_data = data
            try:
                parsed = json.loads(json_data)
                initial_image_url = parsed.get("initial_image_url")
                concept = parsed.get("concept")
                return initial_image_url, concept
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error: {e} - Data: {json_data}")
                return None, None
            
        return None, None
