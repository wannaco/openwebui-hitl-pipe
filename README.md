# HITL Pipe for Open WebUI

## What It Does
A **Pipe function** that adds Human-in-the-Loop confirmation to tool execution in Open WebUI. It:
1. Reads tools from Open WebUI's `__tools__` (whatever you toggle on in the chat UI)
2. Sends tool definitions to any **OpenAI-compatible** LLM endpoint
3. Intercepts every tool call with a **native confirmation dialog**
4. Executes approved tools via the callable Open WebUI provides
5. Renders results using Open WebUI's native collapsible UI
6. Loops until the model returns a final text answer

Works with **any** tool server configured in Open WebUI — MCPO, native MCP, built-in tools, etc. The LLM backend can be **LiteLLM**, **Ollama**, **vLLM**, **OpenAI**, or anything OpenAI-compatible.

## Architecture
```
User → Open WebUI → Pipe (owns the agentic loop)
                       ├── __tools__ (tool specs from Open WebUI)
                       ├── LLM endpoint (POST /v1/chat/completions)
                       ├── __event_call__ (confirmation dialogs)
                       └── Tool execution:
                             ├── callable(**args) — if Open WebUI provides one
                             └── HTTP POST to tool server — MCPO fallback
```

Open WebUI passes tool definitions to the pipe via `__tools__`. The pipe converts specs to OpenAI format, sends them to the LLM, and executes tools either via callable (if provided) or HTTP POST to the tool server.

## Setup
1. Paste `human-in-the-loop-pipe.py` into **Admin → Functions**
2. Configure the LLM valves (URL, API key, model)
3. In a chat, select the pipe as your model
4. **Toggle tools on** using the wrench icon in the chat UI
5. Send a message — the pipe will ask for confirmation before executing tools

## Valves (Configuration)
| Valve | Purpose | Default |
|---|---|---|
| `LITELLM_BASE_URL` | Base URL of any OpenAI-compatible API | `http://localhost:4000` |
| `LITELLM_API_KEY` | API key for the LLM endpoint (leave blank if not needed) | `""` |
| `MODEL_ID` | Model ID to request from the endpoint | `gemini/gemini-2.0-flash` |
| `INJECT_TOOL_SYSTEM_PROMPT` | Inject a system message listing available tools | `True` |
| `REQUEST_TIMEOUT` | HTTP timeout in seconds for LLM calls | `120` |
| `MAX_TOOL_ROUNDS` | Max tool-call rounds before stopping | `10` |
| `AUTO_APPROVE_READ_ONLY` | Auto-approve read-only tools (search, get, list, etc.) | `False` |

### Provider Examples

| Provider | `LITELLM_BASE_URL` | `MODEL_ID` | Notes |
|---|---|---|---|
| **LiteLLM** | `http://localhost:4000` | `gemini/gemini-2.0-flash` | Use LiteLLM model prefixes |
| **Ollama** | `http://localhost:11434` | `qwen2.5:latest` | Ollama exposes `/v1/chat/completions` natively |
| **vLLM** | `http://localhost:8000` | `meta-llama/Llama-3-8B` | vLLM's OpenAI-compat server |
| **OpenAI** | `https://api.openai.com` | `gpt-4o` | Set `LITELLM_API_KEY` to your OpenAI key |
| **LocalAI** | `http://localhost:8080` | `gpt-4` | Follows OpenAI API format |

## Key Technical Details

- **Tool source**: Open WebUI passes `__tools__` to the pipe — two formats exist:
  - `{spec, callable, tool_id, type}` — tools with a direct callable
  - `{spec, direct, server}` — tool-server tools (MCPO) with a server URL
- **Tool execution**: Tries `callable(**args)` first; falls back to HTTP POST to `server.url` using MCPO naming convention (`tool_{path}_{method}` → `POST /{path}`)
- **Native UI rendering**: Yields `<details type="tool_calls">` blocks — Open WebUI renders these as collapsible tool result cards
- **Streaming**: `pipe()` is an async generator (`yield`), not a return
- **Confirmation**: Uses `__event_call__({"type": "confirmation", ...})` — returns truthy if approved, falsy if rejected/dismissed
- **Rejection tracking**: After 2 rejections of the same tool, auto-rejects without prompting and tells the LLM to try a different approach

## Known Limitations
- Clicking outside the confirmation modal = dismiss = rejection (Open WebUI behavior)
- The LLM must support **tool/function calling**
- Tools must be **toggled on** in the chat UI wrench icon — if no tools are enabled, the pipe has nothing to work with
