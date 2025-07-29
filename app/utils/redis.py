from redis.sentinel import Sentinel
from core.config import settings
import json
import uuid
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

        self.stream_key = "original:images"
        self.group_name = "gpu_group"

        start_id = uuid.uuid4().hex

        try:
            self.redis.xgroup_create(
                name = self.stream_key,
                groupname = self.group_name,
                id = start_id,
                mkstream = True
            )
            logger.info(f"Created Redis stream group: {self.group_name} with start ID: {start_id}")
        except self.redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Redis stream group {self.group_name} already exists, skipping creation.")
            else:
                logger.error(f"Error creating Redis stream group: {e}")
                raise
        
        self.consumer_name = start_id

    def push_completed_image(self, image_url: str, generated_image_url: str = None, products: list[dict] = None) -> None:
        prouducts_list = products if isinstance(products, list) else []

        fields = {
            "initial_image_url": image_url,
            "processed_image_url": generated_image_url,
            "products": json.dumps(prouducts_list)
        }

        completed_stream_key = "completed:images"

        new_id = self.redis.xadd(
            name = completed_stream_key,
            fields = fields,
            id = "*",
            maxlen = 1000,
            approximate = True
        )

    def pop_original_image(self):
        resp = self.redis.xreadgroup(
            groupname = self.group_name,
            consumername = self.consumer_name,
            streams = {self.stream_key: ">"},
            count = 1,
            block = 0
        )

        if not resp:
            return None, None
        
        _, messages = resp[0]
        msg_id, fields = messages[0]

        initial_image_url = fields.get("initial_image_url")
        concept = fields.get("concept")

        self.redis.xack(self.stream_key, self.group_name, msg_id)
        self.redis.xdel(self.stream_key, msg_id)

        logger.info(f"Popped image from Redis stream: {initial_image_url}, concept: {concept}")
        return initial_image_url, concept
