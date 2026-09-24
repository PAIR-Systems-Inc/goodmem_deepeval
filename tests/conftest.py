"""Test fixtures.

The SDK is driven through an httpx MockTransport replaying bytes captured
from a live server (v1.0.320), so the real SDK decoders run and nothing about
the wire format is invented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodmem import Goodmem
import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def ndjson_events(name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in load_bytes(name).decode().strip().split("\n")
        if line.strip()
    ]


def to_ndjson(events: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def ndjson_response(events: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        content=to_ndjson(events),
        headers={"Content-Type": "application/x-ndjson; charset=utf-8"},
    )


class Recorder:
    """Route requests to canned responses and remember what was sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[Any] = []
        self._routes: list[tuple[str, str, httpx.Response]] = []

    def route(self, method: str, path: str, response: httpx.Response) -> None:
        self._routes.append((method.upper(), path, response))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            self.bodies.append(json.loads(request.content) if request.content else None)
        except ValueError:
            self.bodies.append(request.content)
        for method, path, response in self._routes:
            if request.method == method and request.url.path == path:
                return response
        raise AssertionError(f"unrouted {request.method} {request.url.path}")

    @property
    def last_body(self) -> Any:
        return self.bodies[-1]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(recorder: Recorder) -> Goodmem:
    """A real SDK client whose transport replays fixtures.

    The SDK refuses base_url/api_key alongside an injected client, so both
    belong on the httpx.Client.
    """
    http_client = httpx.Client(
        transport=recorder.transport(),
        base_url="https://goodmem.test",
        headers={"x-api-key": "gm_test_key_not_a_real_credential"},
    )
    return Goodmem(http_client=http_client)
