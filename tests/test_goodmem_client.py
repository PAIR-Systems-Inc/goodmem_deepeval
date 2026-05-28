"""Unit tests for the GoodMem client transport layer."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from deepeval_goodmem._client import GoodMemClient

BASE_URL = "https://goodmem.test"
API_KEY = "test-api-key"


def _mock_response(*, json_data=None, text=None, headers=None, status_code=200):
    """Build a mock response object that mimics the httpx Response surface."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = json_data
    response.text = text if text is not None else (json.dumps(json_data) if json_data else "")
    response.raise_for_status.return_value = None
    return response


@pytest.fixture()
def mock_httpx_client():
    """Patch httpx.Client inside deepeval_goodmem._client and yield the session mock."""
    with patch("deepeval_goodmem._client.httpx.Client") as client_class:
        session = MagicMock()
        client_class.return_value.__enter__.return_value = session
        client_class.return_value.__exit__.return_value = None
        yield session


@pytest.fixture()
def client():
    """Build a GoodMemClient against a fake base URL."""
    return GoodMemClient(base_url=BASE_URL, api_key=API_KEY)


_NDJSON_WITH_RESULT = "\n".join(
    [
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-1"}}),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c-1",
                            "chunkText": "GoodMem stores memories.",
                            "memoryId": "mem-1",
                        },
                        "relevanceScore": 0.92,
                        "memoryIndex": 0,
                    }
                }
            }
        ),
    ]
)

_NDJSON_EMPTY = json.dumps({"resultSetBoundary": {"resultSetId": "rs-empty"}})


def test_retrieve_metadata_filter_attaches_to_every_space_key(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_WITH_RESULT)
    filter_expr = "CAST(val('$.category') AS TEXT) = 'feat'"

    client.retrieve_memories(
        query="new features",
        space_ids="sp-1,sp-2",
        wait_for_indexing=False,
        metadata_filter=filter_expr,
    )

    body = mock_httpx_client.post.call_args.kwargs["json"]
    assert body["spaceKeys"] == [
        {"spaceId": "sp-1", "filter": filter_expr},
        {"spaceId": "sp-2", "filter": filter_expr},
    ]


def test_retrieve_metadata_filter_none_omits_filter_key(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_WITH_RESULT)

    client.retrieve_memories(
        query="any",
        space_ids="sp-1,sp-2",
        wait_for_indexing=False,
        metadata_filter=None,
    )

    body = mock_httpx_client.post.call_args.kwargs["json"]
    assert body["spaceKeys"] == [{"spaceId": "sp-1"}, {"spaceId": "sp-2"}]


def test_retrieve_metadata_filter_empty_string_omits_filter_key(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_WITH_RESULT)

    client.retrieve_memories(
        query="any",
        space_ids="sp-1",
        wait_for_indexing=False,
        metadata_filter="",
    )

    body = mock_httpx_client.post.call_args.kwargs["json"]
    assert body["spaceKeys"] == [{"spaceId": "sp-1"}]


def test_retrieve_polls_until_max_wait_seconds(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_EMPTY)

    result = client.retrieve_memories(
        query="anything",
        space_ids="sp-1",
        wait_for_indexing=True,
        max_wait_seconds=0.15,
        poll_interval=0.05,
    )

    assert result["totalResults"] == 0
    assert "0.15" in result["message"]
    assert mock_httpx_client.post.call_count >= 2


def test_retrieve_wait_disabled_returns_immediately(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_EMPTY)

    result = client.retrieve_memories(
        query="anything",
        space_ids="sp-1",
        wait_for_indexing=False,
    )

    assert result["totalResults"] == 0
    assert mock_httpx_client.post.call_count == 1


def test_retrieve_default_max_wait_is_sixty_seconds(client):
    import inspect

    sig = inspect.signature(client.retrieve_memories)
    assert sig.parameters["max_wait_seconds"].default == 60.0
    assert sig.parameters["poll_interval"].default == 5.0


def test_list_memories_sort_by_and_sort_order_set(client, mock_httpx_client):
    mock_httpx_client.get.return_value = _mock_response(json_data={"memories": []})

    client.list_memories(
        space_id="sp-1",
        sort_by="created_at",
        sort_order="DESCENDING",
    )

    params = mock_httpx_client.get.call_args.kwargs["params"]
    assert params["sortBy"] == "created_at"
    assert params["sortOrder"] == "DESCENDING"


def test_list_memories_sort_by_none_omits_param(client, mock_httpx_client):
    mock_httpx_client.get.return_value = _mock_response(json_data={"memories": []})

    client.list_memories(space_id="sp-1", sort_by=None, sort_order=None)

    params = mock_httpx_client.get.call_args.kwargs["params"]
    assert "sortBy" not in params
    assert "sortOrder" not in params


def test_list_memories_no_sort_params_by_default(client, mock_httpx_client):
    mock_httpx_client.get.return_value = _mock_response(json_data={"memories": []})

    client.list_memories(space_id="sp-1")

    params = mock_httpx_client.get.call_args.kwargs["params"]
    assert "sortBy" not in params
    assert "sortOrder" not in params


def test_retrieve_returns_chunk_payload_unchanged(client, mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response(text=_NDJSON_WITH_RESULT)

    result = client.retrieve_memories(
        query="anything",
        space_ids="sp-1",
        wait_for_indexing=False,
    )

    assert result["success"] is True
    assert result["totalResults"] == 1
    chunk = result["results"][0]
    assert chunk["chunkId"] == "c-1"
    assert chunk["chunkText"] == "GoodMem stores memories."
    assert chunk["relevanceScore"] == 0.92
