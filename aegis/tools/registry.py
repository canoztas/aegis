"""Registry for tool plugins."""

from typing import Dict, Optional, Type
import importlib

from aegis.tools.base import ToolPlugin


class ToolRegistry:
    """Holds tool plugins by ID."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolPlugin] = {}

    def register(self, tool: ToolPlugin) -> None:
        self._tools[tool.tool_id] = tool

    def register_class(self, tool_cls: Type[ToolPlugin], config: Optional[Dict] = None) -> None:
        self.register(tool_cls(config or {}))

    def get(self, tool_id: str) -> Optional[ToolPlugin]:
        tool = self._tools.get(tool_id)
        if tool:
            return tool
        # Allow dynamic import path fallback
        if "." in tool_id:
            tool = self._load_from_path(tool_id)
            if tool:
                self.register(tool)
                return tool
        return None

    def list_tools(self) -> Dict[str, ToolPlugin]:
        return dict(self._tools)

    def _load_from_path(self, path: str) -> Optional[ToolPlugin]:
        try:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            klass = getattr(module, class_name, None)
            if klass and issubclass(klass, ToolPlugin):
                return klass()
        except Exception:
            return None
        return None


DEFAULT_TOOL_REGISTRY = ToolRegistry()


def register_builtin_tools(registry: ToolRegistry = DEFAULT_TOOL_REGISTRY) -> None:
    from aegis.tools.builtin.regex_tool import RegexTool

    registry.register_class(RegexTool)


register_builtin_tools()
