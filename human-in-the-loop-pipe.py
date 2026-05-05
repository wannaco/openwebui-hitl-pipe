"""
title: Human in the Loop
author: assistant
version: 5.4.0
required_open_webui_version: 0.5.0
description: HITL agentic pipe — uses tools from Open WebUI (toggle in chat UI), calls any OpenAI-compatible LLM, and intercepts every tool call with a confirmation dialog.
"""

import json
import html
import inspect
import base64
import httpx
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator


class Pipe:
    class Valves(BaseModel):
        LITELLM_BASE_URL: str = Field(
            default="http://localhost:4000",
            description="Base URL of any OpenAI-compatible API (LiteLLM, Ollama, vLLM, OpenAI, etc.).",
        )
        LITELLM_API_KEY: str = Field(
            default="",
            description="API key for the LLM endpoint. Leave blank if not needed.",
        )
        MODEL_ID: str = Field(
            default="gemini/gemini-2.0-flash",
            description="Model ID to request from the endpoint.",
        )
        INJECT_TOOL_SYSTEM_PROMPT: bool = Field(
            default=True,
            description="Inject a system message listing available tools to help the model pick the right one.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=120,
            description="HTTP timeout in seconds for LLM calls.",
        )
        MAX_TOOL_ROUNDS: int = Field(
            default=10,
            description="Max tool-call rounds before forcing a text response.",
        )
        AUTO_APPROVE_READ_ONLY: bool = Field(
            default=False,
            description="Auto-approve read-only tools (search, get, list, read, fetch, query).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Build OpenAI tool definitions from __tools__
    # ------------------------------------------------------------------

    def _build_tool_defs(self, tools: dict) -> list[dict]:
        """Convert Open WebUI __tools__ dict to OpenAI function-calling format."""
        openai_tools = []
        for name, tool in tools.items():
            spec = tool.get("spec")
            if not spec:
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": spec.get("name", name),
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self, messages: list, tools: list) -> dict:
        url = f"{self.valves.LITELLM_BASE_URL.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        payload = {
            "model": self.valves.MODEL_ID,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Tool-server HTTP execution (MCPO tools without callable)
    # ------------------------------------------------------------------

    async def _execute_via_server(self, name: str, args: dict, server: dict) -> str:
        """Execute tool via HTTP POST to its tool server (MCPO convention)."""
        base_url = server["url"].rstrip("/")

        # Derive endpoint path from tool name: tool_{path}_{method} → POST /{path}
        # e.g. tool_create_issue_post → POST /create_issue
        path = name
        if path.startswith("tool_"):
            path = path[5:]  # strip "tool_" prefix
        # Strip HTTP method suffix
        for suffix in ("_post", "_get", "_put", "_delete", "_patch"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break

        url = f"{base_url}/{path}"

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=args)
                r.raise_for_status()
                text = r.text
                # Try to parse and clean up JSON response
                try:
                    data = r.json()
                    data = self._decode_base64_fields(data)
                    return json.dumps(data, indent=2, default=str)
                except (json.JSONDecodeError, ValueError):
                    return text
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _decode_base64_fields(self, data):
        """Decode base64 content fields in API responses to prevent double-encoding."""
        if isinstance(data, dict):
            if data.get("encoding") == "base64" and "content" in data:
                try:
                    data["content"] = base64.b64decode(data["content"]).decode("utf-8")
                    data["encoding"] = "plain"
                except (ValueError, UnicodeDecodeError):
                    pass
            # Recurse into nested dicts
            for key, val in data.items():
                if isinstance(val, (dict, list)):
                    data[key] = self._decode_base64_fields(val)
        elif isinstance(data, list):
            data = [self._decode_base64_fields(item) for item in data]
        return data

    # ------------------------------------------------------------------
    # Tool execution via callable
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict, tools: dict) -> str:
        """Call the tool's callable from __tools__ and return result as string."""
        tool = tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})

        # Open WebUI uses different keys depending on tool type
        fn = tool.get("callable")
        if not fn or not callable(fn):
            direct = tool.get("direct")
            if callable(direct):
                fn = direct

        # If we have a callable, invoke it
        if fn and callable(fn):
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = fn(**args)

                # Callable may return a tuple — extract the content
                if isinstance(result, tuple):
                    result = result[0] if result else ""

                if isinstance(result, str):
                    return result
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": f"{type(e).__name__}: {e}"})

        # Fallback: tool-server tools (direct=True, server has url)
        server = tool.get("server")
        if isinstance(server, dict) and server.get("url"):
            return await self._execute_via_server(name, args, server)

        return json.dumps({"error": f"Tool '{name}' has no executable path. Keys: {list(tool.keys())}"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_read_only(self, name: str) -> bool:
        tokens = {"search", "get", "list", "read", "fetch", "query", "find", "lookup", "describe"}
        return any(tok in name.lower() for tok in tokens)

    def _make_details_block(
        self, call_id: str, name: str, arguments: dict, result: str, done: bool = True
    ) -> str:
        return (
            f'<details type="tool_calls" done="{str(done).lower()}" '
            f'id="{html.escape(call_id)}" name="{html.escape(name)}" '
            f'arguments="{html.escape(json.dumps(arguments))}">\'n'
            f'<summary>Tool Executed</summary>\n'
            f'{html.escape(result)}\n'
            f'</details>\n'
        )

    # ------------------------------------------------------------------
    # Main pipe
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __event_call__=None,
        __tools__=None,
    ) -> AsyncGenerator[str, None]:
        try:
            if not __tools__:
                yield "**No tools available.** Toggle tools on in the chat UI (wrench icon) before sending a message."
                return

            # -- Build tool definitions from __tools__ --
            openai_tools = self._build_tool_defs(__tools__)

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"Ready — {len(openai_tools)} tools available", "done": True}}
                )

            # -- Inject system prompt with tool catalog --
            messages = list(body.get("messages", []))

            if self.valves.INJECT_TOOL_SYSTEM_PROMPT and openai_tools:
                tool_lines = []
                for t in openai_tools:
                    tname = t["function"]["name"]
                    tdesc = t["function"].get("description", "")
                    tool_lines.append(f"  - {tname}: {tdesc}" if tdesc else f"  - {tname}")
                catalog = (
                    "You have the following tools available:\n\n"
                    + "\n".join(tool_lines)
                    + "\n\nPick the most appropriate tool for the task."
                )
                messages.insert(0, {"role": "system", "content": catalog})

            # -- Agentic loop --
            rejected_counts: dict[str, int] = {}

            for round_num in range(self.valves.MAX_TOOL_ROUNDS):
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": f"Thinking (round {round_num + 1})…", "done": False}}
                    )

                result = await self._call_llm(messages, openai_tools)
                choice = result.get("choices", [{}])[0]
                msg = choice.get("message", {})
                tool_calls = msg.get("tool_calls", [])

                # No tool calls → yield final text and stop
                if not tool_calls:
                    if __event_emitter__:
                        await __event_emitter__(
                            {"type": "status", "data": {"description": "Done", "done": True}}
                        )
                    content = msg.get("content", "")
                    if content:
                        yield content
                    return

                # -- Confirm each tool call --
                approved = []
                rejected = []

                for tc in tool_calls:
                    fn_info = tc.get("function", {})
                    name = fn_info.get("name", "unknown")
                    try:
                        args = json.loads(fn_info.get("arguments", "{}"))
                        args_pretty = json.dumps(args, indent=2)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                        args_pretty = fn_info.get("arguments", "{}")

                    # Auto-approve read-only
                    if self.valves.AUTO_APPROVE_READ_ONLY and self._is_read_only(name):
                        approved.append((tc, args))
                        continue

                    # Skip tools rejected too many times
                    if rejected_counts.get(name, 0) >= 2:
                        rejected.append((tc, args))
                        continue

                    # Confirmation dialog
                    if __event_call__:
                        ok = await __event_call__(
                            {
                                "type": "confirmation",
                                "data": {
                                    "title": f"Tool: {name}",
                                    "message": f"Allow **{name}**?\n\n```json\n{args_pretty}\n```",
                                },
                            }
                        )
                        if ok:
                            approved.append((tc, args))
                        else:
                            rejected.append((tc, args))
                            rejected_counts[name] = rejected_counts.get(name, 0) + 1
                    else:
                        approved.append((tc, args))

                # Build assistant message with ALL tool_calls (LLM API requires it)
                all_tcs = approved + rejected
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": tc["function"],
                        }
                        for i, (tc, _) in enumerate(all_tcs)
                    ],
                }
                messages.append(assistant_msg)

                # Nothing approved — tell the model firmly
                if not approved:
                    for tc, _ in rejected:
                        rname = tc.get("function", {}).get("name", "")
                        count = rejected_counts.get(rname, 1)
                        if count >= 2:
                            rejection_msg = f"Rejected by user ({count} times). Do NOT call {rname} again. Try a different approach or answer without this tool."
                        else:
                            rejection_msg = "Rejected by user."
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", "call_0"),
                                "content": rejection_msg,
                            }
                        )
                    continue

                # Execute approved tools via callable and render as native UI
                for i, (tc, args) in enumerate(approved):
                    fn_info = tc.get("function", {})
                    name = fn_info.get("name", "unknown")
                    call_id = tc.get("id", f"call_{i}")

                    if __event_emitter__:
                        await __event_emitter__(
                            {"type": "status", "data": {"description": f"Executing: {name}…", "done": False}}
                        )

                    result_text = await self._execute_tool(name, args, __tools__)

                    # Yield native tool result block
                    yield self._make_details_block(call_id, name, args, result_text)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_text,
                        }
                    )

                # Add rejected tool results
                for tc, _ in rejected:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", "rejected"),
                            "content": "Rejected by user.",
                        }
                    )

            # Max rounds exceeded
            yield "Reached maximum tool call rounds. Please try a simpler request."

        except httpx.HTTPStatusError as e:
            yield f"**HTTP Error** ({e.response.status_code}): {e.response.text[:500]}"
        except httpx.ConnectError as e:
            yield f"**Connection Error**: {e}"
        except Exception as e:
            yield f"**Error**: {type(e).__name__}: {e}"
