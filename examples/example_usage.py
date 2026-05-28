"""End-to-end demo of the deepeval-goodmem operations.

Runs three scenarios in sequence: persistent project context, a scribe and
analyst pipeline, and metadata-driven retrieval. Reads ``GOODMEM_API_KEY``,
``GOODMEM_BASE_URL``, ``GOODMEM_VERIFY_SSL``, and ``OPENAI_API_KEY`` from
the environment. Run from the repo root with ``python
examples/example_usage.py``. The answering step uses OpenAI; install
``deepeval-goodmem[examples]`` to pull in the openai client.
"""

import json
import os
import sys
import time
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from deepeval_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteSpace,
    GoodMemListEmbedders,
    GoodMemRetrieveMemories,
)
from openai import OpenAI

INDEXING_WAIT_SECONDS = 6
OPENAI_MODEL = os.environ.get("GOODMEM_DEMO_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "Answer the user's question using only the provided notes. If the notes do not cover the question, say so plainly."
)


def store_text(space_id: str, text: str, metadata: dict | None = None) -> None:
    """Store a plain-text memory in the given space."""
    GoodMemCreateMemory().run(
        space_id=space_id,
        text_content=text,
        metadata=metadata,
    )


def retrieve(
    space_id: str,
    query: str,
    metadata_filter: str | None = None,
    max_results: int = 5,
) -> dict:
    """Run a semantic search against the given space and return the parsed payload."""
    return json.loads(
        GoodMemRetrieveMemories().run(
            query=query,
            space_ids=space_id,
            max_results=max_results,
            metadata_filter=metadata_filter,
            max_wait_seconds=20,
            poll_interval=3,
        )
    )


def answer_with_chunks(question: str, chunks: list[dict], openai_client: OpenAI) -> str:
    """Ask OpenAI to answer the question grounded only in the retrieved chunks."""
    context = "\n".join(f"- {c['chunkText']}" for c in chunks) or "(no notes)"
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Notes:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return completion.choices[0].message.content.strip()


def main() -> None:
    """Run the three demo scenarios end to end."""
    openai_client = OpenAI()

    embedders = json.loads(GoodMemListEmbedders().run())
    embedder_id = embedders["embedders"][0]["embedderId"]

    project_space = json.loads(
        GoodMemCreateSpace().run(
            name="deepeval-goodmem-demo-project",
            embedder_id=embedder_id,
        )
    )["spaceId"]
    team_space = json.loads(
        GoodMemCreateSpace().run(
            name="deepeval-goodmem-demo-team",
            embedder_id=embedder_id,
        )
    )["spaceId"]
    tagged_space = json.loads(
        GoodMemCreateSpace().run(
            name="deepeval-goodmem-demo-tagged",
            embedder_id=embedder_id,
        )
    )["spaceId"]

    try:
        print("\n=== Scenario 1: persistent project context ===")
        project_facts = [
            "I'm building a customer support assistant for our SaaS product.",
            "The team uses Python 3.12 with FastAPI and Postgres.",
            "For tests we use pytest with at least 80% coverage required.",
        ]
        for fact in project_facts:
            store_text(project_space, fact)
            print(f"  stored: {fact}")
        time.sleep(INDEXING_WAIT_SECONDS)

        question = "Remind me what our coverage requirement is."
        results = retrieve(project_space, question)
        print(f"\n  question: {question}")
        print("  answer:   " + answer_with_chunks(question, results["results"], openai_client))

        print("\n=== Scenario 2: two-role team pipeline ===")
        team_notes = [
            "Q2 goal: reduce customer support response time to under 2 hours.",
            "Our main services are auth-service, billing-service, and notifications-service.",
            "Known issue: notifications-service drops messages during high load.",
            "Team retro: the CI pipeline is too slow; we should parallelize tests.",
        ]
        for note in team_notes:
            store_text(team_space, note)
            print(f"  scribe stored: {note}")
        time.sleep(INDEXING_WAIT_SECONDS)

        question = "What do we know about our services and current priorities?"
        results = retrieve(team_space, question)
        print(f"\n  analyst question: {question}")
        print("  analyst answer:   " + answer_with_chunks(question, results["results"], openai_client))

        print("\n=== Scenario 3: metadata-driven retrieval ===")
        tagged_entries = [
            ("Added user profile editing to the dashboard.", "feat"),
            ("Built the CSV export feature.", "feat"),
            ("Resolved slow login on the mobile app.", "fix"),
            ("Fixed crash when opening large attachments.", "fix"),
            ("Upgraded Python version across services.", "chore"),
            ("Updated the API reference for billing endpoints.", "docs"),
        ]
        for content, category in tagged_entries:
            store_text(tagged_space, content, metadata={"category": category})
            print(f"  stored ({category}): {content}")
        time.sleep(INDEXING_WAIT_SECONDS)

        question = "Show me the new features we've shipped."
        results = retrieve(
            tagged_space,
            question,
            metadata_filter="CAST(val('$.category') AS TEXT) = 'feat'",
        )
        print(f"\n  question: {question}")
        for r in results["results"]:
            print(f"  match:    {r['chunkText']}")
        print("  answer:   " + answer_with_chunks(question, results["results"], openai_client))

    finally:
        for space_id in (project_space, team_space, tagged_space):
            GoodMemDeleteSpace().run(space_id=space_id)

    print(
        f"\nDemo complete. Three spaces were created and deleted on "
        f"{os.environ.get('GOODMEM_BASE_URL', 'the GoodMem server')}."
    )


if __name__ == "__main__":
    main()
