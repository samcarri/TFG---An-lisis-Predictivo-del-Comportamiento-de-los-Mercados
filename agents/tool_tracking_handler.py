"""
Tool Tracking Callback Handler

Captura todas las llamadas a tools que hace el agente durante su ejecución.
Registra nombre, input y output de cada tool call.

Uso:
    from agents.tool_tracking_handler import ToolTrackingHandler

    handler = ToolTrackingHandler(verbose=True)
    agent = create_market_agent(callback_handler=handler)
    response = agent("Analiza NVDA")

    print(handler.summary())
    for call in handler.tool_calls:
        print(call)
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class ToolTrackingHandler:
    """
    Callback handler que captura las llamadas a tools del agente.
    Compatible con el protocolo de Strands callback_handler.
    """

    def __init__(self, verbose: bool = False, agent_name: str = "Agent"):
        self.agent_name = agent_name
        self.verbose = verbose
        self.tool_calls: List[Dict] = []
        self._current_tool: Optional[Dict] = None

    def __call__(self, **kwargs: Any) -> None:
        """Recibe eventos del agente Strands."""
        event = kwargs.get("event", {})
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)

        # Texto generado por el modelo
        if data and self.verbose:
            print(data, end="" if not complete else "\n")

        # Inicio de tool call
        tool_use = (
            event.get("contentBlockStart", {})
            .get("start", {})
            .get("toolUse")
        )
        if tool_use:
            self._current_tool = {
                "tool_id": tool_use.get("toolUseId", ""),
                "name": tool_use.get("name", ""),
                "input": {},
                "output": None,
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "duration_ms": None,
            }
            if self.verbose:
                print(f"\n🔧 [{self.agent_name}] Tool: {self._current_tool['name']}")

        # Input de la tool (puede llegar en chunks)
        content_delta = event.get("contentBlockDelta", {}).get("delta", {})
        if content_delta.get("toolUse") and self._current_tool:
            input_chunk = content_delta["toolUse"].get("input", "")
            if input_chunk:
                # Acumular el input JSON
                existing = self._current_tool.get("_input_raw", "")
                self._current_tool["_input_raw"] = existing + input_chunk

        # Fin de bloque de contenido — parsear input acumulado
        if event.get("contentBlockStop") and self._current_tool:
            raw = self._current_tool.pop("_input_raw", "")
            if raw:
                try:
                    self._current_tool["input"] = json.loads(raw)
                except json.JSONDecodeError:
                    self._current_tool["input"] = {"raw": raw}

        # Resultado de la tool
        tool_result = None
        for content_item in event.get("message", {}).get("content", []):
            if "toolResult" in content_item:
                tool_result = content_item["toolResult"]
                break

        if tool_result and self._current_tool:
            tool_id = tool_result.get("toolUseId", "")
            if tool_id == self._current_tool.get("tool_id"):
                finished_at = datetime.now().isoformat()
                started = datetime.fromisoformat(self._current_tool["started_at"])
                finished = datetime.fromisoformat(finished_at)
                duration_ms = int((finished - started).total_seconds() * 1000)

                # Extraer output
                output_content = tool_result.get("content", [])
                if output_content and isinstance(output_content, list):
                    text_parts = [
                        c.get("text", "") for c in output_content if "text" in c
                    ]
                    raw_output = "\n".join(text_parts)
                    try:
                        output = json.loads(raw_output)
                    except (json.JSONDecodeError, ValueError):
                        output = raw_output
                else:
                    output = output_content

                self._current_tool["output"] = output
                self._current_tool["finished_at"] = finished_at
                self._current_tool["duration_ms"] = duration_ms

                self.tool_calls.append(dict(self._current_tool))

                if self.verbose:
                    print(
                        f"✅ [{self.agent_name}] {self._current_tool['name']} "
                        f"({duration_ms}ms)"
                    )

                self._current_tool = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Limpia el historial de tool calls."""
        self.tool_calls = []
        self._current_tool = None

    def get_calls_for_tool(self, tool_name: str) -> List[Dict]:
        """Retorna todas las llamadas a una tool específica."""
        return [c for c in self.tool_calls if c["name"] == tool_name]

    def summary(self) -> str:
        """Resumen legible de todas las tool calls."""
        if not self.tool_calls:
            return f"[{self.agent_name}] No se llamó a ninguna tool."

        lines = [f"[{self.agent_name}] Tool calls ({len(self.tool_calls)}):"]
        for i, call in enumerate(self.tool_calls, 1):
            duration = f"{call['duration_ms']}ms" if call['duration_ms'] else "?"
            input_str = json.dumps(call['input'], ensure_ascii=False)[:80]
            lines.append(f"  {i}. {call['name']} ({duration}) — input: {input_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Exporta el historial completo como dict."""
        return {
            "agent": self.agent_name,
            "total_calls": len(self.tool_calls),
            "tools_used": list({c["name"] for c in self.tool_calls}),
            "calls": self.tool_calls,
        }
