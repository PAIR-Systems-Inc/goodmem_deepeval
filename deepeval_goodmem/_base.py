"""Base class for GoodMem operations."""

import json
import os
from typing import Any

from deepeval_goodmem._client import GoodMemClient


def _env_bool(value: str | None, default: bool = True) -> bool:
    """Parse a boolean from an environment variable string."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class GoodMemOperation:
    """Base class for GoodMem operation wrappers.

    Each subclass exposes a single GoodMem API action and serializes its
    result as a JSON string.

    Args:
        goodmem_base_url: The GoodMem API base URL. Falls back to the
            `GOODMEM_BASE_URL` environment variable.
        goodmem_api_key: The GoodMem API key. Falls back to the
            `GOODMEM_API_KEY` environment variable.
        goodmem_verify_ssl: Whether to verify SSL certificates. Falls back
            to the `GOODMEM_VERIFY_SSL` environment variable (default: True).
    """

    name: str = "goodmem_operation"
    description: str = ""

    def __init__(
        self,
        *,
        goodmem_base_url: str | None = None,
        goodmem_api_key: str | None = None,
        goodmem_verify_ssl: bool | None = None,
    ) -> None:
        base_url = goodmem_base_url or os.environ.get("GOODMEM_BASE_URL")
        api_key = goodmem_api_key or os.environ.get("GOODMEM_API_KEY")

        if not base_url:
            msg = (
                "GoodMem base URL is required. Pass `goodmem_base_url` or set "
                "the GOODMEM_BASE_URL environment variable."
            )
            raise ValueError(msg)
        if not api_key:
            msg = (
                "GoodMem API key is required. Pass `goodmem_api_key` or set "
                "the GOODMEM_API_KEY environment variable."
            )
            raise ValueError(msg)

        if goodmem_verify_ssl is None:
            verify_ssl = _env_bool(os.environ.get("GOODMEM_VERIFY_SSL"), default=True)
        else:
            verify_ssl = goodmem_verify_ssl

        self.goodmem_base_url = base_url
        self.goodmem_api_key = api_key
        self.goodmem_verify_ssl = verify_ssl

    def _client(self) -> GoodMemClient:
        """Return a fresh `GoodMemClient` for this operation."""
        return GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )

    def run(self, **kwargs: Any) -> str:
        """Run the operation and return the result as a JSON string.

        On failure, the JSON payload contains `success: false` and an
        `error` field with the exception message.
        """
        try:
            result = self._run(**kwargs)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Perform the API call. Subclasses must implement this."""
        raise NotImplementedError
