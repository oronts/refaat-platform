from __future__ import annotations

import secrets
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_secret(prefix: str, size: int = 32) -> str:
    return f"{prefix}_{secrets.token_urlsafe(size)}"
