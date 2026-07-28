"""MRRG-facing helpers kept separate for future external-tool integration."""

from .arch import Architecture, RoutingNode, build_architecture

__all__ = ["Architecture", "RoutingNode", "build_architecture"]
