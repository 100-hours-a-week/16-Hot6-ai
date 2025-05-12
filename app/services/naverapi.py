import os
import re
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.config import settings

class NaverAPI:
    def __init__(self, itemlist: List[str]):
        self.itemlist = itemlist
        self.client_id = settings.NAVER_CLIENT_ID
        self.client_secret = settings.NAVER_CLIENT_SECRET

    def clean_html(self, raw_html: str) -> str:
        return re.sub(re.compile('<.*?>'), '', raw_html)

    def extract_categories(self, item: Dict[str, Any]):
        cat2 = item.get("category2", "")
        cat3 = item.get("category3", "")
        cat4 = item.get("category4", "")
        if cat4:
            return cat3, cat4
        elif cat3:
            return cat2, cat3
        else:
            return "", cat2

    def search_item(self, query: str, display: int = 3) -> List[Dict[str, Any]]:
        url = "https://openapi.naver.com/v1/search/shop.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {"query": query, "display": display}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return []

        items = response.json().get("items", [])
        products = []
        for product in items:
            title = self.clean_html(product.get("title", ""))
            mall = product.get("mallName", "Unknown")
            cat3, cat4 = self.extract_categories(product)

            products.append({
                "name": title,
                "price": int(product["lprice"]),
                "purchase_place": mall,
                "purchase_url": product.get("link"),
                "image_path": product.get("image"),
                "main_category": cat3,
                "sub_category": cat4
            })
        return products

    def run(self) -> List[Dict[str, Any]]:
        result = []
        for item in self.itemlist:
            result.extend(self.search_item(item))
        return result