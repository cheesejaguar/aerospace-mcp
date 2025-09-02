# Agent Tools Documentation

The Aerospace MCP server now includes two powerful agent tools that help users interact more effectively with the aerospace calculation tools using AI assistance.

## Overview

These tools use **OpenAI's GPT-5-Medium model** through **LiteLLM** to provide intelligent assistance for:

1. **Data Formatting**: Automatically format user data into the correct structure for aerospace tools
2. **Tool Selection**: Help users choose the right aerospace tool for their specific task

## Configuration

### Environment Variables

The agent tools are controlled by two main environment variables:

#### 1. LLM_TOOLS_ENABLED (Required)
Controls whether the LLM-powered agent tools are enabled:

```bash
# Enable agent tools
export LLM_TOOLS_ENABLED=true

# Disable agent tools (default)
export LLM_TOOLS_ENABLED=false
```

**Default**: `false` (agent tools disabled by default for security and cost control)

#### 2. OPENAI_API_KEY (Required when enabled)
OpenAI API key for LLM access, required only when `LLM_TOOLS_ENABLED=true`:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Get your API key from: https://platform.openai.com/api-keys

### Configuration File Setup

You can also use a `.env` file for configuration. Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Error Handling

The agent tools provide clear error messages for different configuration states:
- **LLM tools disabled**: "Error: LLM agent tools are disabled. Set LLM_TOOLS_ENABLED=true to enable them."
- **Missing API key**: "Error: OPENAI_API_KEY environment variable not set. Cannot use agent tools."
- **Invalid tool**: Lists all available tools when an invalid tool name is provided

## Available Agent Tools

### 1. format_data_for_tool

Helps format data in the correct format for a specific aerospace-mcp tool.

**Parameters:**
- `tool_name` (str): Name of the aerospace-mcp tool to format data for
- `user_requirements` (str): Description of what the user wants to accomplish
- `raw_data` (str, optional): Any raw data that needs to be formatted

**Returns:**
- JSON string with correctly formatted parameters for the specified tool
- Error message if tool not found or API key missing

**Example Usage:**
```python
result = format_data_for_tool(
    tool_name="plan_flight",
    user_requirements="I want to plan a flight from San Francisco to New York using an A320",
    raw_data=""
)
# Returns: {"departure": {"city": "San Francisco"}, "arrival": {"city": "New York"}, "aircraft": {"type": "A320", "cruise_alt_ft": 37000}, "route_options": {}}
```

### 2. select_aerospace_tool

Helps select the most appropriate aerospace-mcp tool for a given task.

**Parameters:**
- `user_task` (str): Description of what the user wants to accomplish
- `user_context` (str, optional): Additional context about the user's situation

**Returns:**
- Detailed recommendation with tool name(s) and usage guidance
- Structured response with PRIMARY_TOOL, WORKFLOW, CONSIDERATIONS, and EXAMPLE sections

**Example Usage:**
```python
result = select_aerospace_tool(
    user_task="I need to calculate the fuel consumption for a transatlantic flight",
    user_context="Flying from London to New York with a Boeing 777"
)
# Returns detailed guidance on which tools to use and how
```

## Supported Aerospace Tools Catalog

The agent tools are aware of these aerospace-mcp tools:

| Tool Name | Description |
|-----------|-------------|
| `search_airports` | Search for airports by IATA code or city name |
| `plan_flight` | Generate complete flight plan between airports |
| `calculate_distance` | Calculate great-circle distance between airports |
| `get_aircraft_performance` | Get performance estimates for aircraft |
| `get_atmosphere_profile` | Calculate atmospheric conditions at various altitudes |
| `wind_model_simple` | Calculate wind profiles at various altitudes |
| `elements_to_state_vector` | Convert orbital elements to state vector |
| `propagate_orbit_j2` | Propagate satellite orbit with J2 perturbations |
| `hohmann_transfer` | Calculate Hohmann transfer orbit between two circular orbits |
| `rocket_3dof_trajectory` | Simulate 3DOF rocket trajectory with atmospheric effects |

## Integration Examples

### Claude Desktop Integration

When using with Claude Desktop, users can leverage these agent tools for enhanced aerospace calculations:

**Configuration in Claude Desktop settings:**
```json
{
  "mcpServers": {
    "aerospace-mcp": {
      "command": "uv",
      "args": ["run", "aerospace-mcp"],
      "cwd": "/path/to/aerospace-mcp",
      "env": {
        "LLM_TOOLS_ENABLED": "true",
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

**Usage Example:**
```
User: "I want to plan a rocket launch to reach 100km altitude"

Assistant uses select_aerospace_tool:
- Analyzes the request
- Recommends rocket_3dof_trajectory tool
- Provides usage guidance and parameter requirements

User: "Help me format the data for the rocket tool"

Assistant uses format_data_for_tool:
- Takes user's requirements
- Returns properly formatted JSON for rocket_3dof_trajectory
```

### Error Handling

The agent tools include robust error handling:

1. **Missing API Key**: Clear error message with setup instructions
2. **Invalid Tool Name**: Lists all available tools
3. **API Failures**: Graceful degradation with error context
4. **Invalid JSON**: Validation and error reporting

### Performance Considerations

- **Default Behavior**: Agent tools are **disabled by default** to prevent unexpected API usage and costs
- **Response Time**: Typically 2-5 seconds per query (when enabled, depends on OpenAI API)
- **Token Usage**: Optimized prompts to minimize token consumption
- **Caching**: No caching implemented - each request is fresh
- **Rate Limits**: Subject to OpenAI API rate limits
- **Cost Control**: Tools only consume API tokens when explicitly enabled via `LLM_TOOLS_ENABLED=true`

## Development Notes

### Architecture

- Built on **FastMCP** framework for easy tool registration
- Uses **LiteLLM** for unified API access to different LLM providers
- **Pydantic** models for type safety and validation
- Modular design allows easy addition of new aerospace tools

### Testing

Comprehensive test suite includes:
- Unit tests for catalog structure
- Error handling validation
- Integration tests (when API key available)
- Tool registration verification

Run tests with:
```bash
pytest tests/test_agent_tools.py -v
```

### Adding New Tools to the Catalog

To add a new aerospace tool to the agent tools catalog, update the `AEROSPACE_TOOLS` list in `agents.py`:

```python
ToolReference(
    name="your_new_tool",
    description="Description of what the tool does",
    parameters={
        "param1": "type - description",
        "param2": "type - description"
    },
    examples=[
        'your_new_tool(param1="value1", param2="value2")'
    ]
)
```

## Limitations

1. **API Dependency**: Requires active OpenAI API key and internet connection
2. **Cost**: Each query consumes OpenAI API tokens
3. **Model Limitations**: Subject to GPT-5-Medium capabilities and knowledge cutoff
4. **No Offline Mode**: Cannot function without API access

## Future Enhancements

Potential improvements for the agent tools:

1. **Local LLM Support**: Add support for local models via Ollama
2. **Caching**: Implement response caching for common queries
3. **Tool Validation**: Automatically validate generated parameters
4. **Interactive Mode**: Multi-turn conversations for complex tasks
5. **Custom Prompts**: Allow users to customize system prompts
6. **Usage Analytics**: Track tool usage and effectiveness metrics

## Support

For issues with the agent tools:

1. Verify `OPENAI_API_KEY` is set correctly
2. Check internet connectivity
3. Review error messages for specific guidance
4. Ensure aerospace-mcp is up to date
5. Run test suite to verify functionality

The agent tools are designed to enhance the aerospace-mcp experience by making complex tools more accessible through natural language interaction.
