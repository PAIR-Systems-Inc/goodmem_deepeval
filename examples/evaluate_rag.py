"""Evaluate a GoodMem-backed retrieval with DeepEval.

    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_API_KEY=gm_… \
    GOODMEM_SPACE_ID=… \
    python examples/evaluate_rag.py

The RAG metrics need an evaluation model (OPENAI_API_KEY or a DeepEval-
configured model). GoodMemRetrievalHealthMetric does not.
"""

from __future__ import annotations

import os

from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric

from deepeval_goodmem import GoodMemRetrievalHealthMetric, GoodMemRetriever

SPACE_ID = os.environ["GOODMEM_SPACE_ID"]
QUESTION = os.getenv("QUESTION", "what is this corpus about?")

retriever = GoodMemRetriever(space_id=SPACE_ID, limit=5)

result = retriever.search(QUESTION)
print(f"score_kind={result['score_kind']}  partial={result['partial']}")
for hit in result["hits"]:
    print(f"  {hit['score']:+.4f}  {hit['chunk_text'][:70]}")
if result["statuses"]:
    print("  server statuses:", [s["code"] for s in result["statuses"]])

# Your generation step goes here; the retrieved chunks are the context.
answer = "\n".join(h["chunk_text"] for h in result["hits"][:1]) or "I don't know."

case = retriever.to_test_case(result, actual_output=answer)

metrics = [GoodMemRetrievalHealthMetric()]
if os.getenv("OPENAI_API_KEY"):
    metrics.append(ContextualRelevancyMetric())

evaluate([case], metrics)
