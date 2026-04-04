from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import (
    REQUIRE_AUTH,
    SUPABASE_ANON_KEY,
    SUPABASE_JWT_ALGORITHM,
    SUPABASE_JWT_SECRET,
    SUPABASE_URL,
)


@dataclass
class CurrentUser:
    id: str
    email: str | None = None
    email_confirmed: bool = False
    raw_claims: dict | None = None


_bearer_scheme = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> dict:
    # Prefer server-side verification against Supabase for newer projects
    # using asymmetric signing keys. Fall back to local JWT secret only when
    # explicitly configured.
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        return _fetch_supabase_user(token)

    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentication is enabled but Supabase verification is not configured."
            ),
        )

    try:
        return jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=[SUPABASE_JWT_ALGORITHM],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired.",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
        ) from exc


def _fetch_supabase_user(token: str) -> dict:
    url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_ANON_KEY,
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify access token with Supabase.",
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach Supabase for token verification.",
        ) from exc

    return {
        "sub": data.get("id"),
        "email": data.get("email"),
        "email_confirmed_at": data.get("email_confirmed_at"),
        "user_metadata": data.get("user_metadata") or {},
        "app_metadata": data.get("app_metadata") or {},
    }


def _build_current_user(claims: dict) -> CurrentUser:
    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated token is missing a user identifier.",
        )

    email_confirmed_at = claims.get("email_confirmed_at")
    return CurrentUser(
        id=user_id,
        email=claims.get("email"),
        email_confirmed=bool(email_confirmed_at),
        raw_claims=claims,
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CurrentUser:
    if not REQUIRE_AUTH:
        return CurrentUser(
            id="dev-user",
            email="dev-user@example.com",
            email_confirmed=True,
            raw_claims={"role": "development"},
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    claims = _decode_token(credentials.credentials)
    user = _build_current_user(claims)
    if not user.email_confirmed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email confirmation is required before using this feature.",
        )
    return user
