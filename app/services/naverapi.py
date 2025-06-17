import re, requests, logging, random
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
        # 1. smartstore URL
        match = re.search(r"/products/(\d+)", url)
        if match:
            return match.group(1)

        # 2. catalog 형식
        match = re.search(r"/catalog/(\d+)", url)
        if match:
            return match.group(1)

        # 3. id= 형식 (드물게 있음)
        match = re.search(r"id=(\d+)", url)
        if match:
            return match.group(1)

        return "xxxxx"

    @lru_cache(maxsize=128)
    def _fetch(self, query: str, display: int) -> List[Dict[str, Any]]:
        """네이버 쇼핑 API 실제 호출 (결과 캐싱)"""
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

            results.append({
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
            })
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
        """
        단일 쿼리 검색 + 페일오버(재시도) 로직
        - 결과가 display 개수보다 적으면, "<query> 소품" 으로 보강 검색
        """
        disp = display or self.DEFAULT_DISPLAY
        results = self._fetch(query, disp)

        if len(results) < disp:
            needed = disp - len(results)
            fallback_q = f"{query}{self.FALLBACK_SUFFIX}"
            logger.info(f"[NaverAPI] fallback retry for '{query}' as '{fallback_q}'")
            results += self._fetch(fallback_q, needed)

        return results

    def aggregate_results(
        self,
        products: List[Dict[str, Any]],
        max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        전체 상품 리스트 중복 제거 및 상위 max_items 개만 남김
        - 상품명(name) 기준으로 최초 등장 순으로 dedupe
        """
        seen = OrderedDict()
        for prod in products:
            if prod["name"] not in seen:
                seen[prod["name"]] = prod
            if len(seen) >= max_items:
                break
        return list(seen.values())


    @staticmethod
    def is_valid_naver_url(url: str) -> bool:
        return any(domain in url for domain in [
            "smartstore.naver.com",
            "search.shopping.naver.com/catalog/"
        ])


    def run(self, total: int = 5, per_prefix: int = 1) -> List[Dict[str, Any]]:
        all_prods = []
        prefixes = self.PREFIXES[self.category]

        for q in self.build_queries():
            batch = self.search_item(q, self.DEFAULT_DISPLAY)
            for p in batch:
                p["_prefix"] = q.split()[0]
            all_prods.extend(batch)

        # ✅ 네이버 쇼핑만 필터링 (옵션 기반)
        if self.run_only_naver:
            filtered_prods = [p for p in all_prods if self.is_valid_naver_url(p["purchase_url"])]
        else:
            filtered_prods = all_prods

        # sub_category별 1개씩
        by_sub = defaultdict(list)
        for p in filtered_prods:
            by_sub[p["sub_category"]].append(p)
        selected = [random.choice(lst) for lst in by_sub.values() if lst]

        # prefix별로 per_prefix 개씩
        by_pref = defaultdict(list)
        for p in filtered_prods:
            by_pref[p["_prefix"]].append(p)
        for pref in prefixes:
            pool = [p for p in by_pref[pref] if p not in selected]
            selected += random.sample(pool, min(per_prefix, len(pool)))

        # ✅ 부족할 경우
        if not self.run_only_naver:
            remaining = [p for p in all_prods if p not in selected]
            need = total - len(selected)
            if need > 0 and remaining:
                selected += random.sample(remaining, min(need, len(remaining)))

        # 정리
        for p in selected:
            p.pop("_prefix", None)
        random.shuffle(selected)

        return selected