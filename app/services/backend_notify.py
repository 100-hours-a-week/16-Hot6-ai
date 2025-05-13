import json, copy, requests
from core.config import settings
import logging

logger = logging.getLogger(__name__)

def notify_backend(image_url: str, generated_image_url: str = None, products: list[dict] = None) -> None:
    backend_url = settings.RESULT_POST_URL

    payload = {
        "initial_image_url": image_url,
        "processed_image_url": generated_image_url,
        "products": products
    }

    try:
        response = requests.post(
            backend_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"[Backend Notify] Response code: {response.status_code}")

        if response.status_code not in (200, 201):
            logger.error(f"[Backend Notify] Failed: {response.status_code}")
        else:
            short_payload = copy.deepcopy(payload)
            if products and len(products) > 2:
                short_payload["products"] = products[:2]
                short_payload["products"].append(f"... ({len(products) - 2} more items)")
            logger.info("[Backend Notify] Payload:\n" + json.dumps(short_payload, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"[Backend Notify] Exception occurred: {e}")