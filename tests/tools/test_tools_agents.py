from __future__ import annotations


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
