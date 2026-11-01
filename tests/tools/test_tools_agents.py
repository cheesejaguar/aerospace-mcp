from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_agents_module_flags():
    """These tests reload ``agents`` under a patched environment.

    ``monkeypatch`` restores the environment but not the module constants the
    reload computed from it, and the registry holds the same module's
    functions, so restore the flags or later tests see LLM tools enabled.
    """
    import aerospace_mcp.tools.agents as agents

    saved = {
        name: getattr(agents, name) for name in ("LLM_TOOLS_ENABLED", "_AGENT_MODEL")
    }
    yield
    for name, value in saved.items():
        setattr(agents, name, value)


def test_agents_errors(monkeypatch):
    import importlib

    # Disabled tools
    monkeypatch.setenv("LLM_TOOLS_ENABLED", "false")
    import aerospace_mcp.tools.agents as agents

    importlib.reload(agents)
    format_data_for_tool = agents.format_data_for_tool
    select_aerospace_tool = agents.select_aerospace_tool

    assert (
        "disabled" in format_data_for_tool("search_airports", "plan a flight").lower()
    )
    assert "disabled" in select_aerospace_tool("plan a flight").lower()

    # Enabled but missing key
    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    import importlib

    import aerospace_mcp.tools.agents as agents2

    importlib.reload(agents2)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "not set" in agents2.format_data_for_tool("search_airports", "plan").lower()
    assert "not set" in agents2.select_aerospace_tool("plan").lower()


def _llm_returning(content: str, calls: list | None = None):
    """Stand-in for the LiteLLM client (the paid LLM-provider boundary).

    Records each ``completion`` request in ``calls`` and answers with
    ``content`` in the OpenAI response shape LiteLLM returns.
    """

    class Msg:
        pass

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    Choice.message.content = content

    class StubLLM:
        def completion(self, *a, **k):
            if calls is not None:
                calls.append(k)
            return Resp()

    return StubLLM()


def _enable(monkeypatch):
    import importlib

    import aerospace_mcp.tools.agents as agents

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    importlib.reload(agents)
    return agents


def test_agents_success(monkeypatch):
    agents = _enable(monkeypatch)
    calls = []
    agents.litellm = _llm_returning('  {"query": "San Jose", "country": "US"}\n', calls)  # type: ignore

    out = agents.format_data_for_tool(
        "search_airports", "Find airports in San Jose", "country=US"
    )

    # The model's JSON is returned stripped and unchanged
    assert out == '{"query": "San Jose", "country": "US"}'
    # The request describes the chosen tool and the user's input
    (request,) = calls
    assert request["model"] == "gpt-5.6-sol"
    assert request["temperature"] == 0.1
    system, user = request["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    ref = next(t for t in agents.AEROSPACE_TOOLS if t.name == "search_airports")
    assert ref.description in system["content"]
    assert "Raw Data (if provided): country=US" in system["content"]
    assert (
        user["content"] == "Format data for search_airports: Find airports in San Jose"
    )


def test_format_data_unknown_tool(monkeypatch):
    agents = _enable(monkeypatch)
    calls = []
    agents.litellm = _llm_returning("{}", calls)  # type: ignore

    out = agents.format_data_for_tool("no_such_tool", "anything")
    assert out.startswith("Error: Tool 'no_such_tool' not found")
    assert "search_airports" in out
    assert calls == []  # no LLM call for an unknown tool


def test_agent_model_default_and_override(monkeypatch):
    calls = []

    monkeypatch.delenv("LLM_MODEL", raising=False)
    agents = _enable(monkeypatch)
    agents.litellm = _llm_returning('{"query": "SJC"}', calls)  # type: ignore
    agents.format_data_for_tool("search_airports", "Find airports")

    monkeypatch.setenv("LLM_MODEL", "custom/provider-model")
    agents = _enable(monkeypatch)
    agents.litellm = _llm_returning('{"query": "SJC"}', calls)  # type: ignore
    agents.format_data_for_tool("search_airports", "Find airports")

    assert [c["model"] for c in calls] == ["gpt-5.6-sol", "custom/provider-model"]


def test_agents_exception_paths(monkeypatch):
    import json

    agents = _enable(monkeypatch)

    class BadLLM:
        def completion(self, *a, **k):
            raise RuntimeError('provider said "no"')

    agents.litellm = BadLLM()  # type: ignore
    out = agents.format_data_for_tool("search_airports", "task")
    # A JSON error object, even when the message contains quotes
    assert json.loads(out) == {"error": 'Failed to format data: provider said "no"'}
    assert agents.select_aerospace_tool("task") == (
        'Error: Failed to select appropriate tool: provider said "no"'
    )


def test_select_tool_success(monkeypatch):
    agents = _enable(monkeypatch)
    calls = []
    agents.litellm = _llm_returning(
        "PRIMARY_TOOL: search_airports\nWORKFLOW: ...\n\n", calls
    )  # type: ignore

    out = agents.select_aerospace_tool("Find airports in San Jose", "for a trip")

    assert out == "PRIMARY_TOOL: search_airports\nWORKFLOW: ..."
    (request,) = calls
    assert request["temperature"] == 0.2
    system, user = request["messages"]
    # Every tool in the catalogue is offered to the model
    for tool in agents.AEROSPACE_TOOLS:
        assert f'"name": "{tool.name}"' in system["content"]
    assert user["content"] == (
        "Help me with this task: Find airports in San Jose\n\nContext: for a trip"
    )


def test_format_data_invalid_json(monkeypatch):
    import json

    agents = _enable(monkeypatch)
    agents.litellm = _llm_returning('not "json"')  # type: ignore

    out = agents.format_data_for_tool("search_airports", "Find airports")
    # Used to embed the raw text unescaped, producing invalid JSON
    assert json.loads(out) == {
        "error": 'Failed to generate valid JSON format. Raw response: not "json"'
    }


def test_server_import_does_not_import_litellm():
    """LiteLLM costs ~2.5 s of startup and the agent tools are off by default."""
    import subprocess
    import sys

    code = (
        "import sys, aerospace_mcp.fastmcp_server\n"
        "assert 'litellm' not in sys.modules, 'litellm imported at startup'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_litellm_is_imported_and_configured_on_first_use(monkeypatch):
    import importlib
    import sys
    import types

    import aerospace_mcp.tools.agents as agents

    importlib.reload(agents)
    assert agents.litellm is None
    fake = types.ModuleType("litellm")
    fake.set_verbose = True
    monkeypatch.setitem(sys.modules, "litellm", fake)

    assert agents._get_litellm() is fake
    assert fake.set_verbose is False
    assert agents.litellm is fake
    # A stub assigned by a caller (as the tests above do) is used as is.
    stub = object()
    agents.litellm = stub
    assert agents._get_litellm() is stub


def test_missing_litellm_reports_the_llm_extra(monkeypatch):
    """LiteLLM is the optional 'llm' extra; without it the tools say so."""
    import importlib
    import sys

    import aerospace_mcp.tools.agents as agents

    importlib.reload(agents)
    monkeypatch.setattr(agents, "LLM_TOOLS_ENABLED", True)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "litellm", None)  # import raises ImportError

    with pytest.raises(ImportError, match=r"aerospace-mcp\[llm\]"):
        agents._get_litellm()
    out = agents.format_data_for_tool("search_airports", "Find airports")
    assert "aerospace-mcp[llm]" in out
