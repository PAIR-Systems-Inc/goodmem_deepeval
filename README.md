# deepeval-goodmem

GoodMem retrieval for [DeepEval](https://deepeval.com).

Every retrieval is a DeepEval **retriever span**, and a retrieval turns
directly into an `LLMTestCase` with `retrieval_context` populated — which is
what `ContextualPrecisionMetric`, `ContextualRecallMetric`,
`ContextualRelevancyMetric` and `FaithfulnessMetric` actually read.

```bash
pip install deepeval-goodmem
```

## Retrieve and evaluate

```python
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, ContextualRecallMetric
from deepeval_goodmem import GoodMemRetriever, GoodMemRetrievalHealthMetric

retriever = GoodMemRetriever(space_name="docs", limit=5)   # credentials from the environment

result = retriever.search("how do I rotate an API key?")
answer = my_llm(result["hits"])                            # your generation step

case = retriever.to_test_case(result, actual_output=answer)

evaluate([case], [
    ContextualRelevancyMetric(),
    ContextualRecallMetric(),
    GoodMemRetrievalHealthMetric(),
])
```

`search()` is decorated with `@observe(type="retriever")` and reports `top_k`
and the embedder through `update_retriever_span`, so the call appears as a
retriever span with its query, latency and results.

If you only need the context, `retrieve()` returns the chunk text directly:

```python
context = retriever.retrieve("how do I rotate an API key?")   # list[str]
```

### What `search()` returns

| Key | What it is |
| --- | --- |
| `hits` | `chunk_id`, `chunk_text`, `memory_id`, `space_id`, `source`, `score`, `score_kind`, `metadata` — in the server's order |
| `score_kind` | `"vector"` or `"reranker"`. Different scales; see below |
| `statuses` | Statuses indicating a real problem, `[]` when clean |
| `partial` | `True` when the server reported a real problem during this retrieval, with or without hits |
| `abstract_reply` | The server-generated summary, only when `llm_id` is set |
| `space_ids` | Which spaces were actually searched |

A retrieval that reported a problem and returned nothing is an empty `hits`
with `partial: True` **and a warning** — never an exception, and never
indistinguishable from "no matches".

## The health metric

```python
from deepeval_goodmem import GoodMemRetrievalHealthMetric

evaluate(cases, [ContextualRecallMetric(), GoodMemRetrievalHealthMetric()])
```

Every other RAG metric scores the *content* that came back, so a broken
reranker and a thin corpus look identical: both are poor recall.
`GoodMemRetrievalHealthMetric` reads the diagnostics `to_test_case` puts in
`metadata["goodmem"]` and scores the retrieval itself — `1.0` clean, `0.0`
degraded, with the server's status codes in `reason` and `score_breakdown`.
No LLM, so it costs nothing and never flakes. A test case with no GoodMem
metadata is skipped rather than failed.

## Scores

GoodMem returns two different things in the same field:

| | Range observed live | Best match is |
| --- | --- | --- |
| Vector score | negative, e.g. `-0.57` | the **lowest** number |
| Reranker score | can also go negative | the **highest** number |

So results keep **the server's order** and are never re-sorted; `score_kind`
says which scale you have; and `min_score` is applied client-side only when
`reranker_id` is set. The server's `relevance_threshold` is never sent.

Even with a reranker the scale is **model-dependent**: on the same documents
Voyage `rerank-2.5` scored `0.27..0.93` and Jina `jina-reranker-v3` scored
`-0.14..0.43`. A `min_score` tuned for one empties the other, so when a
threshold removes every hit the retriever warns and names the observed range.
Calibrate `min_score` for the reranker you use; there is no default.

## Filtering

```python
retriever = GoodMemRetriever(space_name="docs", metadata_filter={"category": "billing"})
retriever.search("refunds", metadata_filter={"lang": "en"})     # AND-ed per call
```

Values are quoted for the GoodMem filter grammar (backslash escaping, verified
against a live server; control characters refused). For anything more complex,
pass an expression directly with `filter=...`.

## Attaching to a space by name

```python
GoodMemRetriever(space_name="docs", embedder_id="…")                     # reuse or fail
GoodMemRetriever(space_name="docs", embedder_id="…", create_space=True)  # or create it
```

Attach-by-name is idempotent reuse: an existing space whose embedder matches
is reused; a **different** embedder is an error, because retrieving across
mismatched embedders returns plausible-looking nonsense. An ambiguous name is
an error too.

## Credentials

From `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`, or as constructor keywords:

```python
GoodMemRetriever(space_name="docs", base_url="https://localhost:8080",
                 api_key="gm_…", verify_ssl=False)
```

They are deliberately **not** model fields, so they cannot reach a
`model_dump()`, a `repr()` or a trace payload.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && mypy && pytest -m "not integration"
```

The offline suite replays NDJSON captured from a live GoodMem server
(v1.0.320) through the real SDK decoders, so the wire format is never
invented. The live suite needs a server and is skipped without one:

```bash
GOODMEM_BASE_URL=… GOODMEM_API_KEY=… GOODMEM_EMBEDDER_ID=… \
  pytest -m integration
```

There is no default credential anywhere in this repository.

## License

MIT
