"""
Tests para agents/tool_tracking_handler.py

Verifica que el handler captura correctamente las tool calls
sin necesitar Ollama ni APIs externas.
"""

import json
import pytest
from agents.tool_tracking_handler import ToolTrackingHandler


def _fire_tool_start(handler: ToolTrackingHandler, tool_id: str, tool_name: str):
    """Simula el evento de inicio de una tool call."""
    handler(event={
        "contentBlockStart": {
            "start": {
                "toolUse": {
                    "toolUseId": tool_id,
                    "name": tool_name,
                }
            }
        }
    })


def _fire_tool_input(handler: ToolTrackingHandler, input_dict: dict):
    """Simula el evento de input de una tool call."""
    handler(event={
        "contentBlockDelta": {
            "delta": {
                "toolUse": {
                    "input": json.dumps(input_dict)
                }
            }
        }
    })


def _fire_content_block_stop(handler: ToolTrackingHandler):
    """Simula el fin del bloque de contenido (parsea el input acumulado)."""
    handler(event={"contentBlockStop": True})


def _fire_tool_result(handler: ToolTrackingHandler, tool_id: str, output: dict):
    """Simula el evento de resultado de una tool call."""
    handler(event={
        "message": {
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "content": [{"text": json.dumps(output)}],
                    }
                }
            ]
        }
    })


def _simulate_tool_call(handler, tool_id, tool_name, input_dict, output_dict):
    """Simula una tool call completa."""
    _fire_tool_start(handler, tool_id, tool_name)
    _fire_tool_input(handler, input_dict)
    _fire_content_block_stop(handler)
    _fire_tool_result(handler, tool_id, output_dict)


class TestToolTrackingHandler:

    def test_captures_single_tool_call(self):
        """Captura una tool call completa con input y output."""
        handler = ToolTrackingHandler(agent_name="Market")
        _simulate_tool_call(
            handler,
            tool_id="tool-001",
            tool_name="get_market_data",
            input_dict={"ticker": "NVDA", "end_date": "2024-08-07"},
            output_dict={"current_price": 125.4, "trend": "bullish"},
        )

        assert len(handler.tool_calls) == 1
        call = handler.tool_calls[0]
        assert call["name"] == "get_market_data"
        assert call["input"]["ticker"] == "NVDA"
        assert call["input"]["end_date"] == "2024-08-07"
        assert call["output"]["current_price"] == 125.4

    def test_captures_multiple_tool_calls(self):
        """Captura múltiples tool calls en secuencia."""
        handler = ToolTrackingHandler(agent_name="Market")

        _simulate_tool_call(handler, "t1", "get_market_data",
                            {"ticker": "NVDA"}, {"price": 100})
        _simulate_tool_call(handler, "t2", "get_technical_analysis",
                            {"ticker": "NVDA"}, {"trend": "bullish"})
        _simulate_tool_call(handler, "t3", "get_market_calendar_info",
                            {"date": "2024-08-10"}, {"is_trading_day": False})

        assert len(handler.tool_calls) == 3
        assert handler.tool_calls[0]["name"] == "get_market_data"
        assert handler.tool_calls[1]["name"] == "get_technical_analysis"
        assert handler.tool_calls[2]["name"] == "get_market_calendar_info"

    def test_records_duration(self):
        """Registra la duración de cada tool call."""
        handler = ToolTrackingHandler()
        _simulate_tool_call(handler, "t1", "get_market_data", {}, {})

        assert handler.tool_calls[0]["duration_ms"] is not None
        assert handler.tool_calls[0]["duration_ms"] >= 0

    def test_records_timestamps(self):
        """Registra started_at y finished_at."""
        handler = ToolTrackingHandler()
        _simulate_tool_call(handler, "t1", "get_market_data", {}, {})

        call = handler.tool_calls[0]
        assert call["started_at"] is not None
        assert call["finished_at"] is not None
        assert call["finished_at"] >= call["started_at"]

    def test_get_calls_for_tool(self):
        """Filtra calls por nombre de tool."""
        handler = ToolTrackingHandler()
        _simulate_tool_call(handler, "t1", "get_market_data", {"ticker": "NVDA"}, {})
        _simulate_tool_call(handler, "t2", "get_market_data", {"ticker": "AAPL"}, {})
        _simulate_tool_call(handler, "t3", "get_technical_analysis", {}, {})

        market_calls = handler.get_calls_for_tool("get_market_data")
        assert len(market_calls) == 2
        assert all(c["name"] == "get_market_data" for c in market_calls)

    def test_summary_format(self):
        """summary() retorna un string legible."""
        handler = ToolTrackingHandler(agent_name="TestAgent")
        _simulate_tool_call(handler, "t1", "get_market_data",
                            {"ticker": "NVDA"}, {"price": 100})

        summary = handler.summary()
        assert "TestAgent" in summary
        assert "get_market_data" in summary
        assert "1" in summary

    def test_summary_empty(self):
        """summary() sin calls retorna mensaje apropiado."""
        handler = ToolTrackingHandler(agent_name="TestAgent")
        summary = handler.summary()
        assert "No se llamó" in summary

    def test_to_dict(self):
        """to_dict() retorna estructura completa."""
        handler = ToolTrackingHandler(agent_name="Market")
        _simulate_tool_call(handler, "t1", "get_market_data", {}, {})
        _simulate_tool_call(handler, "t2", "get_technical_analysis", {}, {})

        d = handler.to_dict()
        assert d["agent"] == "Market"
        assert d["total_calls"] == 2
        assert "get_market_data" in d["tools_used"]
        assert "get_technical_analysis" in d["tools_used"]
        assert len(d["calls"]) == 2

    def test_reset(self):
        """reset() limpia el historial."""
        handler = ToolTrackingHandler()
        _simulate_tool_call(handler, "t1", "get_market_data", {}, {})
        assert len(handler.tool_calls) == 1

        handler.reset()
        assert len(handler.tool_calls) == 0

    def test_ignores_unmatched_tool_result(self):
        """No registra resultados de tools que no se iniciaron."""
        handler = ToolTrackingHandler()
        # Resultado sin tool start previo
        _fire_tool_result(handler, "unknown-id", {"data": "something"})
        assert len(handler.tool_calls) == 0

    def test_text_output_captured(self):
        """El texto del modelo se emite sin errores."""
        handler = ToolTrackingHandler(verbose=False)
        # No debe lanzar excepción
        handler(data="Analizando NVDA...", complete=False)
        handler(data="Análisis completado.", complete=True)
        assert len(handler.tool_calls) == 0
