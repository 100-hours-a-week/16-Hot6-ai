import re, json, random, logging
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import requests
from core.config import settings

logger = logging.getLogger(__name__)
_rng = random.Random()          # 스레드끼리 충돌 방지용 로컬 RNG

class NaverAPI:
    BASE_URL = "https://openapi.naver.com/v1/search/shop.json"
    FALLBACK_SUFFIX = " 소품"
    ALLOWED_DOMAINS = [
        "smartstore.naver.com",
        "search.shopping.naver.com/catalog/",
    ]

    PREFIXES = {
        "upgrade": ["ergonomic", "adjustable", "durable"],
        "decor":   ["mini", "wooden", "ceramic"],
    }

    def __init__(
        self,
        raw_items: List[str],
        category: str = "decor",
        sample_pool: int = 10,
        per_keyword: int = 3,
    ):
        """raw_items 예: ["데스크 매트", "데스크 램프"]"""
        self.raw_items   = raw_items
        self.category    = category if category in self.PREFIXES else "decor"
        self.sample_pool = sample_pool
        self.per_keyword = per_keyword

        self._session = requests.Session()
        self.client_id     = settings.NAVER_CLIENT_ID
        self.client_secret = settings.NAVER_CLIENT_SECRET

    # ───────────────────────────────────────────────
    @classmethod
    def is_allowed(cls, url: str) -> bool:
        return any(dom in url for dom in cls.ALLOWED_DOMAINS)

    @staticmethod
    def extract_product_code(url: str) -> str:
        for pat in (r"/products/(\d+)", r"/catalog/(\d+)", r"id=(\d+)"):
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return "xxxxx"

    # ───────────────────────────────────────────────
    def build_queries(self) -> List[str]:
        """접두어 조합 쿼리 생성 (현재 코드에선 사용처 없음)"""
        pref = self.PREFIXES[self.category]
        return [f"{p} {item}" for item in self.raw_items for p in pref]

    # ───────────────────────────────────────────────
    @lru_cache(maxsize=256)
    def _fetch(self, query: str, display: int) -> List[Dict[str, Any]]:
        headers = {
            "X-Naver-Client-Id":     self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        try:
            resp = self._session.get(
                self.BASE_URL, headers=headers,
                params={"query": query, "display": display}, timeout=5
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            logger.error(f"[NaverAPI] request failed for '{query}': {e}")
            return []

        clean: List[Dict[str, Any]] = []
        for prod in items:
            title = re.sub(r"<.*?>", "", prod.get("title", ""))
            main_c, sub_c = self._extract_categories(prod)
            record = {
                "name":           title,
                "price":          int(prod.get("lprice", 0)),
                "purchase_place": prod.get("mallName", "Unknown"),
                "purchase_url":   prod.get("link"),
                "image_path":     prod.get("image"),
                "main_category":  main_c,
                "sub_category":   sub_c,
                "center_x":       None,
                "center_y":       None,
                "product_code":   self.extract_product_code(prod.get("link", "")),
            }
            clean.append(record)
        return clean

    def _extract_categories(self, item: Dict[str, Any]) -> Tuple[str, str]:
        c2, c3, c4 = item.get("category2", ""), item.get("category3", ""), item.get("category4", "")
        if c4:  return c3, c4
        if c3:  return c2, c3
        return "", c2

    # ───────────────────────────────────────────────
    def search_item(self, query: str) -> List[Dict[str, Any]]:
        """sample_pool 개까지 결과 확보, 부족하면 fallback 한 번 더"""
        results = self._fetch(query, self.sample_pool)
        if len(results) < self.sample_pool:
            extra = self._fetch(f"{query}{self.FALLBACK_SUFFIX}", self.sample_pool - len(results))
            results += extra
        return results

    # ───────────────────────────────────────────────
    def run_with_coords(
        self,
        products_with_coords: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        final: List[Dict[str, Any]] = []

        for prod in products_with_coords:
            query = prod.get("name_ko")
            if not query:
                logger.warning(f"[NaverAPI] Missing keyword in {prod}")
                continue

            pool = [r for r in self.search_item(query) if self.is_allowed(r["purchase_url"])]
            if not pool:
                logger.info(f"[NaverAPI] No allowed hit for '{query}'")
                continue

            _rng.shuffle(pool)
            for item in pool[: self.per_keyword]:
                item.update(
                    name=item.get("name", query),
                    center_x=prod.get("center_x"),
                    center_y=prod.get("center_y"),
                )
                final.append(item)

        logger.debug("[NaverAPI] Matched JSON:\n" + json.dumps(final, ensure_ascii=False, indent=2))
        logger.info(f"[NaverAPI] Matched {len(final)} products for {len(products_with_coords)} keywords")
        return final