"""Pure helpers for talking to the Noah API (no LiveKit imports, so they are unit-testable)."""

import os

ALLOWED_LANGUAGES = {"nl", "en", "de", "fr", "es", "tr"}
DEFAULT_LANGUAGE = "nl"

SERVICE_TOKEN_HEADER = "X-Service-Token"


def normalize_language(value: str | None) -> str:
    """Return a supported ISO language code, falling back to Dutch."""
    code = (value or DEFAULT_LANGUAGE).strip().lower()
    return code if code in ALLOWED_LANGUAGES else DEFAULT_LANGUAGE


def get_api_url() -> str:
    """Base URL of the API, e.g. https://.../api/v1 (no trailing slash)."""
    api_url = os.getenv("API_URL") or os.getenv("ELDERLY_COMPANION_API", "")
    if not api_url:
        raise ValueError("API_URL environment variable is not set")
    return api_url.rstrip("/")


def api_headers(extra: dict | None = None) -> dict:
    """Headers for every API request: JSON + the service-to-service token.

    The API rejects all non-public routes without either a user JWT or
    `X-Service-Token` matching its SERVICE_API_TOKEN.
    """
    headers = dict(extra or {})
    headers["Content-Type"] = "application/json"
    token = os.getenv("SERVICE_API_TOKEN")
    if token:
        headers[SERVICE_TOKEN_HEADER] = token
    return headers


def warn_if_service_token_missing() -> None:
    if not os.getenv("SERVICE_API_TOKEN"):
        print(
            "[Config] WARNING: SERVICE_API_TOKEN is not set — all API calls "
            "(user lookup, memory vault, wellbeing, care signals) will fail with 401"
        )


def family_member_elderly_id(user: dict) -> str | None:
    """Elderly user id a family-member record is linked to.

    The API stores this as `linkedElderlyId`; `userId` is accepted for
    backwards compatibility with older payloads.
    """
    return user.get("linkedElderlyId") or user.get("userId")
