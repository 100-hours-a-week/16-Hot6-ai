# fastapi_project/tests/unit/test_services.py
import pytest, asyncio
from app.services import vision2text

@pytest.mark.asyncio
async def test_describe_image_mocked(monkeypatch):
    async def fake(*args, **kwargs): return "테스트 프롬프트"
    monkeypatch.setattr(vision2text, "describe_image", fake)
    result = await vision2text.describe_image("http://example.com")
    assert result == "테스트 프롬프트"
