"""The README's Python examples run as written.

Every ```python block is executed in order, in one namespace, against the
real SDK over the captured-bytes transport. Only the README's own
placeholders are supplied: ``my_llm`` (its "your generation step") and
``cases``. ``deepeval.evaluate`` is replaced by a check that every metric
passed to it has the test-case fields it requires, which is exactly what the
real ``evaluate`` refuses at run time -- so an example that DeepEval would
reject fails here without an LLM call.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import deepeval
from goodmem import Goodmem
import httpx
import pytest

from deepeval_goodmem import _connection
from tests.conftest import Recorder, load_json, ndjson_events, ndjson_response

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


def _require_fields(test_cases: list[Any], metrics: list[Any], **_: Any) -> None:
    for metric in metrics:
        required = getattr(metric, "_required_params", None)
        if not isinstance(required, list):
            continue
        for case in test_cases:
            missing = [p.value for p in required if getattr(case, p.value) is None]
            assert not missing, (
                f"{type(metric).__name__} requires {missing} on the test case; "
                "DeepEval's evaluate() raises MissingTestCaseParamsError"
            )


def test_readme_has_python_examples() -> None:
    assert len(BLOCKS) >= 5


def test_readme_python_examples_run(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = load_json("spaces_list.json")["spaces"][0]
    space["name"] = "docs"
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": [space]}))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson_response(ndjson_events("retrieve_vector.ndjson")),
    )

    def sdk_client(**_: Any) -> Goodmem:
        # A fresh client per session: the connection closes the one it opened.
        return Goodmem(
            http_client=httpx.Client(
                transport=recorder.transport(), base_url="https://goodmem.test"
            )
        )

    monkeypatch.setattr(_connection, "Goodmem", sdk_client)
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://goodmem.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "readme-test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "readme-test-placeholder")
    monkeypatch.setattr(deepeval, "evaluate", _require_fields)

    ns: dict[str, Any] = {"my_llm": lambda hits: hits[0]["chunk_text"] if hits else ""}
    for number, source in enumerate(BLOCKS, 1):
        if "cases" in source and "cases" not in ns and "case" in ns:
            ns["cases"] = [ns["case"]]
        # Executing this repository's own README is the point of the test.
        code = compile(source, f"README.md python block {number}", "exec")
        exec(code, ns)  # noqa: S102

    assert recorder.requests, "the examples never reached the server"
