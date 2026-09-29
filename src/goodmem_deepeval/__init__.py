"""GoodMem for DeepEval.

Retrieval from GoodMem, traced as a DeepEval retriever span and evaluable
with DeepEval's RAG metrics::

    from deepeval import evaluate
    from deepeval.metrics import ContextualRelevancyMetric
    from goodmem_deepeval import GoodMemRetriever, GoodMemRetrievalHealthMetric

    retriever = GoodMemRetriever(space_name="docs")
    result = retriever.search("how do I rotate an API key?")
    case = retriever.to_test_case(result, actual_output=answer)

    evaluate([case], [ContextualRelevancyMetric(),
                      GoodMemRetrievalHealthMetric()])

Credentials come from ``GOODMEM_BASE_URL`` / ``GOODMEM_API_KEY`` or from
constructor keywords, and are never model fields, so they cannot reach a
trace payload or a ``repr()``.
"""

from goodmem_deepeval._connection import GoodMemConnection
from goodmem_deepeval._spaces import GoodMemSpaceError
from goodmem_deepeval.metrics import GoodMemRetrievalHealthMetric
from goodmem_deepeval.retriever import GoodMemRetriever

__version__ = "0.3.0"

__all__ = [
    "GoodMemConnection",
    "GoodMemRetrievalHealthMetric",
    "GoodMemRetriever",
    "GoodMemSpaceError",
    "__version__",
]
