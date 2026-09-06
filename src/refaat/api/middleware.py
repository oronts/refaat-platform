from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from refaat.errors import RateLimitError
from refaat.observability import trace_context


def _canonical_ip(raw: str) -> str | None:
    """Canonical string form of an IP address, or None if it is not a valid IP.

    Canonicalizing collapses spelling variants (an IPv6 address has many textual
    forms) to one key, so an attacker cannot mint rotating rate-limit identities by
    re-spelling the same address, and a malformed entry is rejected rather than
    trusted as a key.
    """
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        return None


def client_ip(scope: Scope, trusted_proxies: Sequence[str]) -> str:
    """Resolve the trusted client IP for rate limiting.

    The socket peer is authoritative. Only when that peer is a configured trusted
    proxy is ``X-Forwarded-For`` consulted, and then the right-most entry that is a
    valid IP and not itself a trusted proxy is used, so a client cannot forge its
    rate-limit identity by prepending spoofed XFF entries. All comparisons run on the
    canonical IP form, and every ``x-forwarded-for`` header is read (duplicates are
    not collapsed) so a spoofed entry in a second header cannot hide behind the first.
    """
    client = scope.get("client")
    peer_raw = str(client[0]) if client else "unknown"
    peer = _canonical_ip(peer_raw) or peer_raw
    trusted = {_canonical_ip(proxy) or proxy for proxy in trusted_proxies}
    if peer not in trusted:
        return peer
    candidates: list[str] = []
    for name, value in scope.get("headers", []):
        if name != b"x-forwarded-for":
            continue
        decoded = value.decode(errors="ignore") if isinstance(value, bytes) else str(value)
        for entry in decoded.split(","):
            canonical = _canonical_ip(entry)
            if canonical is not None:
                candidates.append(canonical)
    for entry in reversed(candidates):
        if entry not in trusted:
            return entry
    return peer


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_length = dict(scope.get("headers", [])).get(b"content-length")
        try:
            declared_size = int(content_length) if content_length else 0
        except ValueError:
            declared_size = self.max_body_bytes + 1
        if declared_size > self.max_body_bytes:
            await self._reject(send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise BodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLargeError:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"error":{"code":"payload_too_large","message":"Request body is too large"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


class BodyTooLargeError(Exception):
    pass


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        requested = headers.get(b"x-request-id", b"").decode()[:64] or None
        with trace_context(trace_id=requested) as trace_id:

            async def send_with_headers(message: Message) -> None:
                if message["type"] == "http.response.start":
                    response_headers = MutableHeaders(scope=message)
                    response_headers["x-request-id"] = trace_id
                    response_headers["x-content-type-options"] = "nosniff"
                    response_headers["referrer-policy"] = "no-referrer"
                    response_headers["permissions-policy"] = "camera=(), microphone=(), geolocation=()"
                await send(message)

            await self.app(scope, receive, send_with_headers)


class InMemoryRateLimiter:
    def __init__(self, requests: int, window_seconds: int, *, max_buckets: int = 10_000) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self.max_buckets = max_buckets
        self.values: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def consume(self, key: str) -> bool:
        now = time.monotonic()
        threshold = now - self.window_seconds
        with self.lock:
            bucket = self.values.get(key)
            if bucket is not None:
                while bucket and bucket[0] <= threshold:
                    bucket.popleft()
                if not bucket:
                    del self.values[key]
                    bucket = None
            if bucket is None:
                if len(self.values) >= self.max_buckets:
                    self._sweep_stale(threshold)
                    if len(self.values) >= self.max_buckets:
                        # Every tracked bucket still has an active window. Evicting one
                        # would reset its history and let that address cycle past the
                        # limit, so refuse the new address instead of evicting an active
                        # bucket.
                        return False
                bucket = deque()
                self.values[key] = bucket
            if len(bucket) >= self.requests:
                return False
            bucket.append(now)
            return True

    def _sweep_stale(self, threshold: float) -> None:
        stale = [key for key, bucket in self.values.items() if not bucket or bucket[-1] <= threshold]
        for key in stale:
            del self.values[key]

    def close(self) -> None:
        self.values.clear()


class RedisRateLimiter:
    def __init__(self, url: str, requests: int, window_seconds: int) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install refaat[redis] for Redis rate limiting") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.requests = requests
        self.window_seconds = window_seconds

    def consume(self, key: str) -> bool:
        bucket = int(time.time() // self.window_seconds)
        redis_key = f"refaat:rate:{key}:{bucket}"
        with self.client.pipeline() as pipeline:
            pipeline.incr(redis_key)
            pipeline.expire(redis_key, self.window_seconds + 2)
            count, _ = pipeline.execute()
        return int(count) <= self.requests

    def close(self) -> None:
        self.client.close()


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, limiter: Any, trusted_proxies: Sequence[str] = ()) -> None:
        self.app = app
        self.limiter = limiter
        self.trusted_proxies = tuple(trusted_proxies)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path", "").startswith("/health"):
            await self.app(scope, receive, send)
            return
        key_value = client_ip(scope, self.trusted_proxies)
        if not self.limiter.consume(key_value):
            body = b'{"error":{"code":"rate_limited","message":"Rate limit exceeded"}}'
            await send(
                {
                    "type": "http.response.start",
                    "status": RateLimitError.status_code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (
                            b"retry-after",
                            str(getattr(self.limiter, "window_seconds", 60)).encode(),
                        ),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
