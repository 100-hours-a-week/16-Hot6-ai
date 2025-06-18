import re, requests, logging
from functools import lru_cache
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any
from core.config import settings

logger = logging.getLogger(__name__)

class NaverAPI:
    BASE_URL = "https://openapi.naver.com/v1/search/shop.json"
    DEFAULT_DISPLAY = 3
    FALLBACK_SUFFIX = " 소품"
    PREFIXES = {
        "upgrade": ["ergonomic", "adjustable", "durable"],
        "decor":   ["mini", "wooden", "ceramic"]
    }

    def __init__(self, raw_items: List[str], category: str = "decor", run_only_naver: bool = True):
        """
        raw_items: ["데스크 매트", "데스크 램프"] 같은 원본 키워드 리스트
        category: "upgrade" 또는 "decor"
        """
        self.raw_items = raw_items
        self.category = category if category in self.PREFIXES else "decor"
        self.run_only_naver = run_only_naver
        self.client_id = settings.NAVER_CLIENT_ID
        self.client_secret = settings.NAVER_CLIENT_SECRET

    def build_queries(self) -> List[str]:
        """raw_items 에 접두어를 붙여서 여러 쿼리 생성"""
        prefixes = self.PREFIXES[self.category]
        return [f"{p} {item}" for item in self.raw_items for p in prefixes]
    
    @staticmethod
    def extract_product_code(url: str) -> str:
        for pattern in [r"/products/(\d+)", r"/catalog/(\d+)", r"id=(\d+)"]:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return "xxxxx"

    def _fetch(self, query: str, display: int) -> List[Dict[str, Any]]:
        headers = {
            "X-Naver-Client-Id":     self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {"query": query, "display": display}

        try:
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=5)
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            logger.error(f"[NaverAPI] request failed for '{query}': {e}")
            return []

        results: List[Dict[str, Any]] = []
        for prod in items:
            title = re.sub(re.compile("<.*?>"), "", prod.get("title", ""))
            mall  = prod.get("mallName", "Unknown")
            price = int(prod.get("lprice", 0))
            main_cat, sub_cat = self._extract_categories(prod)

            result = {
                "name":           title,
                "price":          price,
                "purchase_place": mall,
                "purchase_url":   prod.get("link"),
                "image_path":     prod.get("image"),
                "main_category":  main_cat,
                "sub_category":   sub_cat,
                "center_x":       None,
                "center_y":       None,
                "product_code":   self.extract_product_code(prod.get("link", ""))
            }

            if self.run_only_naver and not self.is_valid_naver_url(result["purchase_url"]):
                continue
            results.append(result)

        return results

    def _extract_categories(self, item: Dict[str, Any]) -> (str, str):
        """category2~4 중 가장 세부 카테고리 2개 리턴"""
        c2 = item.get("category2", "")
        c3 = item.get("category3", "")
        c4 = item.get("category4", "")
        if c4:
            return c3, c4
        if c3:
            return c2, c3
        return "", c2

    def search_item(self, query: str, display: int = None) -> List[Dict[str, Any]]:
        disp = display or self.DEFAULT_DISPLAY
        results = self._fetch(query, disp)
        if len(results) < disp:
            needed = disp - len(results)
            fallback_q = f"{query}{self.FALLBACK_SUFFIX}"
            logger.info(f"[NaverAPI] fallback retry for '{query}' as '{fallback_q}'")
            results += self._fetch(fallback_q, needed)
        return results

    @staticmethod
    def is_valid_naver_url(url: str) -> bool:
        return any(domain in url for domain in [
            "smartstore.naver.com",
            "search.shopping.naver.com/catalog/"
        ])
    
    def run_with_coords(self, products_with_coords: List[Dict[str, Any]], per_keyword: int = 3) -> List[Dict[str, Any]]:
        """
        products_with_coords: [{"name_ko": ..., "dino_label": ..., "center_x": ..., "center_y": ...}, ...]
        → 각 keyword에 대해 네이버 검색 후 가장 적합한 상품 N개 반환 (좌표 포함)
        """
        final_results = []

        for product in products_with_coords:
            query = product.get("name_ko")
            center_x = product.get("center_x")
            center_y = product.get("center_y")
            dino_label = product.get("dino_label")

            if not query:
                logger.warning(f"[NaverAPI] Missing keyword for product: {product}")
                continue

            results = self.search_item(query, display=10)
            if not results:
                logger.warning(f"[NaverAPI] No result found for keyword: {query}")
                continue

            valid_results = [r for r in results if self.is_valid_naver_url(r["purchase_url"])]
            for item in valid_results[:per_keyword]:
                item["center_x"] = center_x
                item["center_y"] = center_y
                item["dino_label"] = dino_label
                item["name"] = item.get("name", query)
                item["product_code"] = self.extract_product_code(item.get("purchase_url", ""))
                final_results.append(item)

        return final_results