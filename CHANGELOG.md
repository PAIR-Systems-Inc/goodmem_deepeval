# Changelog

## 0.2.1

Documentation only; no code change.

### Fixed

- **README: the headline example did not run.** It built the test case with
  `to_test_case(result, actual_output=answer)` and then evaluated it with
  `ContextualRecallMetric`, which requires `expected_output`. Executed as
  written, `evaluate()` stopped with `MissingTestCaseParamsError: 'expected_output'
  cannot be None for the 'Contextual Recall' metric` (both the first example
  and the health-metric example). The example now passes `expected_output`,
  the health-metric example imports what it uses, and the README says which
  RAG metrics need a reference answer. Measured: 4/6 README Python blocks ran
  before, 6/6 after.
- README: the `search()` return table now lists the `query` key it returns.

### Added

- `tests/test_readme.py` executes every README Python block through the real
  SDK over the captured-bytes transport and fails if a metric the README
  passes to `evaluate()` lacks a test-case field it requires. 36 offline
  tests (was 34).

## 0.2.0

A rewrite. 0.1.0 shipped no DeepEval integration: it was fourteen classes
wrapping REST endpoints with `httpx`, and `deepeval` was not even a
dependency. 0.2.0 is what the name promises, built on the official `goodmem`
SDK.

Every claim below was reproduced against a live GoodMem server (v1.0.320)
before the fix, and the offline tests replay bytes captured from it.

### Added

- **`GoodMemRetriever`** — `search()` is decorated with
  `@observe(type="retriever")` and reports `top_k`/embedder through
  `update_retriever_span`, so a retrieval is a DeepEval retriever span.
- **`retrieve()`** — returns `list[str]`, the shape `retrieval_context` wants.
- **`to_test_case()`** — builds an `LLMTestCase` with `retrieval_context`
  populated and the retrieval's diagnostics in `metadata["goodmem"]`.
- **`GoodMemRetrievalHealthMetric`** — a real `BaseMetric` that fails a case
  whose retrieval was degraded, so a broken reranker is its own failure
  instead of a mysterious drop in someone else's recall number.
- 34 offline tests over captured server bytes and 12 live tests with verified
  teardown. 0.1.0 had 0 tests in CI, and no CI.

### Fixed

| Behaviour | 0.1.0 (reproduced live) | 0.2.0 |
| --- | --- | --- |
| Retrieval with a broken reranker | `success: true`, `totalResults: 1` — the server sent `NOT_FOUND` **and** `RERANKING_FAILED` and the NDJSON parser had no `status` branch, so both were dropped. The result was an *unreranked* fallback presented as a successful reranked search | `partial: true` with both statuses; the fallback chunks are kept |
| Retrieval that failed outright | `success: true, totalResults: 0` with a message blaming indexing | Empty `hits`, `partial: true`, statuses, and a warning — never raised |
| Searching an empty space | **63.8 seconds** (`wait_for_indexing=True`, `max_wait_seconds=60`, `poll_interval=5` → 12 re-POSTs, each re-invoking the LLM when `llm_id` was set) | **0.3 seconds**, one request |
| Relevance scores | Documented as "Minimum relevance score (0-1)"; live values were `-0.57` (vector) and `0.17` (reranker) | Reported as sent, in the server's order, with `score_kind`; `min_score` is client-side and reranker-only |
| `metadata_filter` | A raw string pasted into the request. Live, `x' OR '1'='1` subverted the filter and returned a row it should not have | A dict, quoted for the live filter grammar; control characters refused |
| Any error | `run()` caught **everything** and returned `{"success": false, "error": "Client error '400 Bad Request' … check https://developer.mozilla.org/…"}` as a *string* — the server's actual reason (`Invalid space ID format: not-a-uuid`) was discarded for an MDN link | Errors propagate with the server's message |
| `update_space(public_read=…)` | HTTP 400 — the field was removed from the API | Gone |
| Space name collisions | Not checked | Reuse when the embedder matches; a mismatch or an ambiguous name is an error |
| API key | A plain attribute on every operation object | A private connection; never in `model_dump()`, `repr()` or a trace |

### Migration

The fourteen `GoodMem*` operation classes are gone; they were a REST client,
and the `goodmem` SDK does that job better.

```python
# 0.1.0
from deepeval_goodmem import GoodMemRetrieveMemories
out = json.loads(GoodMemRetrieveMemories(...).run(query="…", space_ids=sid))
for r in out["results"]:
    print(r["chunkText"], r["relevanceScore"])

# 0.2.0
from deepeval_goodmem import GoodMemRetriever
result = GoodMemRetriever(space_id=sid).search("…")
for h in result["hits"]:
    print(h["chunk_text"], h["score"], h["score_kind"])

# 0.2.0 — for plain API access, use the SDK directly
from goodmem import Goodmem
Goodmem(base_url=…, api_key=…).spaces.create(...)
```

| 0.1.0 | 0.2.0 |
| --- | --- |
| `GoodMemRetrieveMemories(...).run(query=, space_ids=)` | `GoodMemRetriever(space_ids=[...]).search(query)` |
| `out["results"][i]["chunkText"]` | `result["hits"][i]["chunk_text"]` |
| `out["results"][i]["relevanceScore"]` | `result["hits"][i]["score"]` + `["score_kind"]` |
| `relevance_threshold=` | `min_score=`, with `reranker_id` set |
| `metadata_filter="CAST(...)"` | `metadata_filter={"field": "value"}` (or `filter=` for an expression) |
| `GoodMemCreateSpace` / `GoodMemGetSpace` / … | `goodmem.Goodmem().spaces.*` |
| `GoodMemCreateMemory` / `GoodMemListMemories` / … | `goodmem.Goodmem().memories.*` |

Prior art: the in-tree integration in the `deepeval-goodmem` fork of upstream
DeepEval established the `@observe(type="retriever")` + `update_retriever_span`
shape, the `list[str]` return for `retrieval_context`, and a
`wait_for_indexing` default of `False`. All three are carried over here.

## 0.1.0

Initial release.
