from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str],
    json: object,
    retries: int = 2,
    base_delay: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """POST with bounded exponential-backoff retry on transient provider failures.

    Retries a rate-limit or transient server status and a transport error up to
    `retries` times, then raises the last httpx error so the caller can wrap it. A
    non-retryable status (for example 400 or 401) is raised on the first attempt.
    """

    attempt = 0
    while True:
        try:
            response = client.post(url, headers=dict(headers), json=json)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in RETRYABLE_STATUS_CODES and attempt < retries:
                sleep(base_delay * (2**attempt))
                attempt += 1
                continue
            raise
        except httpx.TransportError:
            if attempt < retries:
                sleep(base_delay * (2**attempt))
                attempt += 1
                continue
            raise
