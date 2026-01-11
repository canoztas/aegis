"""Provider wrapper for tool plugins."""

import asyncio
from typing import Any, Dict, Optional

from aegis.tools import DEFAULT_TOOL_REGISTRY
from aegis.models.schema import ParserResult


class ToolProvider:
    """Adapter for tool plugins to the provider interface."""

    def __init__(self, tool_id: str, tool_config: Optional[Dict[str, Any]] = None):
        self.tool_id = tool_id
        self.tool_config = tool_config or {}
        self.tool = DEFAULT_TOOL_REGISTRY.get(tool_id)
        if not self.tool:
            raise ValueError(f"Tool '{tool_id}' not found")

    def _analyze_sync(self, prompt: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ParserResult:
        context = context or {}
        code = context.get("code", prompt)
        cfg = dict(self.tool_config)
        cfg.update(kwargs)
        return self.tool.analyze_snippet(code, context, cfg)

    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ParserResult:
        return await asyncio.to_thread(self._analyze_sync, prompt, context, **kwargs)

    def generate(self, prompt: str, **kwargs) -> ParserResult:
        return self._analyze_sync(prompt, {"code": prompt, "file_path": "unknown", "line_start": 1}, **kwargs)
