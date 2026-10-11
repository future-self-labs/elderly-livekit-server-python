import pytest

from lib.api_client import (
    SERVICE_TOKEN_HEADER,
    api_headers,
    family_member_elderly_id,
    get_api_url,
    normalize_language,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("nl", "nl"),
        ("EN", "en"),
        ("  de ", "de"),
        ("tr", "tr"),
        (None, "nl"),
        ("", "nl"),
        ("xx", "nl"),
        ("en-US", "nl"),
    ],
)
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


def test_api_headers_include_service_token(monkeypatch):
    monkeypatch.setenv("SERVICE_API_TOKEN", "test-token")
    headers = api_headers({"X-Extra": "1"})
    assert headers[SERVICE_TOKEN_HEADER] == "test-token"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Extra"] == "1"


def test_api_headers_without_token(monkeypatch):
    monkeypatch.delenv("SERVICE_API_TOKEN", raising=False)
    assert SERVICE_TOKEN_HEADER not in api_headers()


def test_api_headers_does_not_mutate_input(monkeypatch):
    monkeypatch.setenv("SERVICE_API_TOKEN", "t")
    extra = {"A": "b"}
    api_headers(extra)
    assert extra == {"A": "b"}


def test_get_api_url(monkeypatch):
    monkeypatch.setenv("API_URL", "https://api.example.com/api/v1/")
    assert get_api_url() == "https://api.example.com/api/v1"
    monkeypatch.delenv("API_URL")
    monkeypatch.delenv("ELDERLY_COMPANION_API", raising=False)
    with pytest.raises(ValueError):
        get_api_url()


def test_family_member_elderly_id():
    assert family_member_elderly_id({"linkedElderlyId": "abc"}) == "abc"
    assert family_member_elderly_id({"userId": "legacy"}) == "legacy"
    assert family_member_elderly_id({}) is None
