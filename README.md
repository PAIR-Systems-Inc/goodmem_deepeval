# deepeval-goodmem

[GoodMem](https://goodmem.ai) integration for
[DeepEval](https://deepeval.com).

GoodMem gives AI agents retrieval-augmented generation (RAG) memory. Store
documents in a space and GoodMem chunks, embeds, and indexes them so your
agent can pull back the most relevant passages on any question.

This package exposes the GoodMem API as a set of DeepEval-flavoured operation
classes. Each operation has a `run(...)` method that returns a JSON string,
making them callable from any evaluation script, test harness, or scoring
pipeline.

## Installation

```bash
pip install deepeval-goodmem
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import json
import os

os.environ["GOODMEM_BASE_URL"] = "https://localhost:8080"
os.environ["GOODMEM_API_KEY"] = "gm_xxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["GOODMEM_VERIFY_SSL"] = "false"  # self-signed local server

from deepeval_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemListEmbedders,
    GoodMemRetrieveMemories,
)

embedders = json.loads(GoodMemListEmbedders().run())
embedder_id = embedders["embedders"][0]["embedderId"]

space = json.loads(
    GoodMemCreateSpace().run(name="quickstart", embedder_id=embedder_id)
)
space_id = space["spaceId"]

GoodMemCreateMemory().run(
    space_id=space_id,
    text_content="The capital of France is Paris.",
)

results = json.loads(
    GoodMemRetrieveMemories().run(
        query="What is the capital of France?",
        space_ids=space_id,
        max_results=3,
    )
)
print(results)
```

Every operation's `run(...)` returns a JSON string. On success the parsed
object includes `"success": true` and operation-specific fields. On failure
it includes `"success": false` and an `"error"` field.

The local GoodMem dev server ships with TLS even on localhost. Set
`GOODMEM_VERIFY_SSL=false` to accept the self-signed certificate during
development; production deployments behind a trusted certificate should
leave it at the default of `true`.

## Available operations

| Operation | Description |
|---|---|
| `GoodMemListEmbedders` | List embedder models available on the server |
| `GoodMemListSpaces` | List all spaces accessible to the API key |
| `GoodMemGetSpace` | Fetch a space by ID |
| `GoodMemCreateSpace` | Create a space (idempotent by name) |
| `GoodMemUpdateSpace` | Update a space's name, labels, or public-read flag |
| `GoodMemDeleteSpace` | Delete a space and all of its memories |
| `GoodMemCreateMemory` | Store text or a file as a memory |
| `GoodMemListMemories` | List memories in a space, with pagination and filters |
| `GoodMemRetrieveMemories` | Semantic retrieval across one or more spaces |
| `GoodMemGetMemory` | Fetch a memory by ID, with optional content |
| `GoodMemDeleteMemory` | Delete a memory |

### Retrieval options

`GoodMemRetrieveMemories` accepts the following parameters in addition to
`query`, `space_ids`, and `max_results`:

| Parameter | Type | Description |
|---|---|---|
| `metadata_filter` | str | SQL-style JSONPath filter applied server-side to every space key. Example: `CAST(val('$.category') AS TEXT) = 'feat'` |
| `wait_for_indexing` | bool | Poll for results when none come back on the first call (default `True`) |
| `max_wait_seconds` | float | Polling budget (default `60.0`) |
| `poll_interval` | float | Seconds between polls (default `5.0`) |
| `reranker_id` | str | Reranker model to refine result ordering |
| `llm_id` | str | LLM that generates a contextual abstract reply |
| `relevance_threshold` | float | Minimum score (0-1) for inclusion |
| `llm_temperature` | float | Creativity (0-2) for the LLM post-processor |
| `chronological_resort` | bool | Reorder results by creation time |

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server |
| `GOODMEM_API_KEY` | API key sent as `X-API-Key` |
| `GOODMEM_VERIFY_SSL` | Set to `false` to skip TLS verification (default `true`) |

When the env vars are set, every operation constructor can be called with
no arguments.

## End-to-end example

[`examples/example_usage.py`](examples/example_usage.py) walks through three
scenarios: persistent project context, a scribe and analyst pipeline, and
metadata-driven retrieval. The answering step uses OpenAI; install with
`pip install deepeval-goodmem[examples]` and set `OPENAI_API_KEY` before
running.

```bash
python examples/example_usage.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
