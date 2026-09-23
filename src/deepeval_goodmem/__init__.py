"""GoodMem for DeepEval.

Retrieval from GoodMem, traced as a DeepEval retriever span and evaluable
with DeepEval's RAG metrics::

    from deepeval import evaluate
    from deepeval.metrics import ContextualRelevancyMetric
    from deepeval_goodmem import GoodMemRetriever, GoodMemRetrievalHealthMetric

    retriever = GoodMemRetriever(space_name="docs")
    result = retriever.search("how do I rotate an API key?")
    case = retriever.to_test_case(result, actual_output=answer)

    evaluate([case], [ContextualRelevancyMetric(),
                      GoodMemRetrievalHealthMetric()])

Credentials come from ``GOODMEM_BASE_URL`` / ``GOODMEM_API_KEY`` or from
constructor keywords, and are never model fields, so they cannot reach a
trace payload or a ``repr()``.
"""

from deepeval_goodmem._connection import GoodMemConnection
from deepeval_goodmem._spaces import GoodMemSpaceError
from deepeval_goodmem.metrics import GoodMemRetrievalHealthMetric
from deepeval_goodmem.retriever import GoodMemRetriever

__version__ = "0.2.0"

__all__ = [
    "GoodMemConnection",
    "GoodMemRetrievalHealthMetric",
    "GoodMemRetriever",
    "GoodMemSpaceError",
    "__version__",
]
