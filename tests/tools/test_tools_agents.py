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


def test_agents_success(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class Choice:
        class Msg:
            content = '{"tool": "search_airports"}'

        message = Msg()

    class Resp:
        choices = [Choice()]

    # Patch litellm.completion
    import aerospace_mcp.tools.agents as agents

    importlib.reload(agents)

    class StubLLM:
        def completion(self, *a, **k):
            return Resp()

    agents.litellm = StubLLM()  # type: ignore

    out = agents.format_data_for_tool("search_airports", "Find airports in San Jose")
    assert "search_airports" in out


def test_agent_model_default_and_override(monkeypatch):
    import importlib

    import aerospace_mcp.tools.agents as agents

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class Choice:
        class Msg:
            content = '{"tool": "search_airports"}'

        message = Msg()

    class Resp:
        choices = [Choice()]

    used_models = []

    class StubLLM:
        def completion(self, *a, **k):
            used_models.append(k["model"])
            return Resp()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    importlib.reload(agents)
    agents.litellm = StubLLM()  # type: ignore
    agents.format_data_for_tool("search_airports", "Find airports")

    monkeypatch.setenv("LLM_MODEL", "custom/provider-model")
    importlib.reload(agents)
    agents.litellm = StubLLM()  # type: ignore
    agents.format_data_for_tool("search_airports", "Find airports")

    assert used_models == ["gpt-5.6-sol", "custom/provider-model"]


def test_agents_exception_paths(monkeypatch):
    import importlib

    import aerospace_mcp.tools.agents as agents

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    importlib.reload(agents)

    class BadLLM:
        def completion(self, *a, **k):
            raise RuntimeError("boom")

    agents.litellm = BadLLM()  # type: ignore
    assert "error" in agents.format_data_for_tool("search_airports", "task").lower()
    assert "error" in agents.select_aerospace_tool("task").lower()


def test_select_tool_success(monkeypatch):
    import importlib

    import aerospace_mcp.tools.agents as agents

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    importlib.reload(agents)

    class Choice:
        class Msg:
            content = "PRIMARY_TOOL: search_airports\\nWORKFLOW: ..."

        message = Msg()

    class Resp:
        choices = [Choice()]

    class StubLLM:
        def completion(self, *a, **k):
            return Resp()

    agents.litellm = StubLLM()  # type: ignore
    out = agents.select_aerospace_tool("Find airports in San Jose")
    assert "PRIMARY_TOOL" in out


def test_format_data_invalid_json(monkeypatch):
    import importlib

    import aerospace_mcp.tools.agents as agents

    monkeypatch.setenv("LLM_TOOLS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    importlib.reload(agents)

    class Choice:
        class Msg:
            content = "not json"

        message = Msg()

    class Resp:
        choices = [Choice()]

    class StubLLM:
        def completion(self, *a, **k):
            return Resp()

    agents.litellm = StubLLM()  # type: ignore
    out = agents.format_data_for_tool("search_airports", "Find airports")
    assert "failed to generate valid json" in out.lower()


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
