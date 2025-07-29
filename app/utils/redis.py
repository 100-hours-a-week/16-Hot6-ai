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
            socket_timeout=10,
            password=redis_password,
            sentinel_kwargs={"password": redis_password}
        )

        self.redis = sentinel.master_for(
            settings.REDIS_MASTER_NAME,
            socket_timeout=60,
            password=redis_password,
            decode_responses=True
        )

    def push_completed_image(self, image_url: str, generated_image_url: str = None, products: list[dict] = None) -> None:
        payload = {
            "initial_image_url": image_url,
            "processed_image_url": generated_image_url,
            "products": products
        }

        fields = {
            "initial_image_url": payload["initial_image_url"],
            "processed_image_url": payload.get("processed_image_url"),
            "products": json.dumps(payload["products"]) if payload.get("products") is not None else ""
        }
        
        self.redis.xadd(
            name="completed:images",
            fields=fields,
            id="*",
            maxlen=1000,
            approximate=True
        )

    def pop_original_image(self):
        entries = self.redis.xread({"original:images": "0"}, count=1, block=0)
        if entries:
            _, messages = entries[0]
            msg_id, fields = messages[0]

            data = {
                (k.decode() if isinstance(k, (bytes, bytearray)) else k):
                (v.decode() if isinstance(v, (bytes, bytearray)) else v)
                for k, v in fields.items()
            }

            logger.info(f"Pop original image: {msg_id}")
            return data.get("initial_image_url"), data.get("concept")
        # data = self.redis.blpop("original:images", timeout=0)
        # if data:
        #     _, json_data = data
        #     try:
        #         parsed = json.loads(json_data)
        #         initial_image_url = parsed.get("initial_image_url")
        #         concept = parsed.get("concept")
        #         return initial_image_url, concept
        #     except json.JSONDecodeError as e:
        #         logger.error(f"JSON Decode Error: {e} - Data: {json_data}")
        #         return None, None
            
        # return None, None
