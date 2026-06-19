"""Human-readable help formatter for backend capabilities.

Usage from CLI::

    help backends              -- list all registered backends
    help backend milvus        -- detailed info for one backend

Usage from Python::

    from benchmark.backends._help import format_backend_help, format_backends_list
    print(format_backends_list(registry))
    print(format_backend_help(registry, "milvus"))
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BackendRegistry
    from .base import BackendDescriptor, IndexDescriptor


def format_backends_list(reg: "BackendRegistry") -> str:
    """One-line summary of every registered backend."""
    backends = reg.list_backends()
    if not backends:
        return "No backends registered."

    lines = ["Registered vector-database backends:", ""]
    name_width = max(len(d.display_name) for d in backends)
    for desc in backends:
        first_line = desc.description.split(".")[0].strip() + "."
        metrics = ", ".join(desc.supported_metrics)
        indexes = ", ".join(desc.index_names())
        lines.append(
            f"  {desc.display_name:<{name_width}}  "
            f"(name: {desc.name})"
        )
        lines.append(
            f"  {'':<{name_width}}  "
            f"metrics: {metrics}"
        )
        lines.append(
            f"  {'':<{name_width}}  "
            f"indexes: {indexes}"
        )
        lines.append("")

    lines.append(
        "Use 'help backend <name>' for detailed parameters.  "
        "Example: help backend milvus"
    )
    return "\n".join(lines)


def format_backend_help(reg: "BackendRegistry", name: str) -> str:
    """Detailed help for one backend, including every parameter."""
    desc = reg.get(name)
    if desc is None:
        available = ", ".join(reg.names()) or "(none)"
        return f"Unknown backend '{name}'.  Available: {available}"
    return _render_descriptor(desc)


# ------------------------------------------------------------------
# Internal renderers
# ------------------------------------------------------------------

_SEPARATOR = "-" * 64


def _render_descriptor(desc: "BackendDescriptor") -> str:
    parts: list[str] = []

    # Header
    parts.append("=" * 64)
    parts.append(f"Backend: {desc.display_name}  (--backend {desc.name})")
    parts.append("=" * 64)
    parts.append("")
    parts.append(textwrap.fill(desc.description, width=64))
    parts.append("")

    # Metrics
    parts.append("Supported distance metrics:")
    for m in desc.supported_metrics:
        parts.append(f"  - {m}")
    parts.append("")

    # Connection params
    if desc.connection_params:
        parts.append(_SEPARATOR)
        parts.append("Connection parameters:")
        parts.append(_SEPARATOR)
        parts.append("")
        for p in desc.connection_params:
            parts.append(_render_param(p))
        parts.append("")

    # Index types
    if desc.supported_indexes:
        parts.append(_SEPARATOR)
        parts.append("Index types:")
        parts.append(_SEPARATOR)
        for idx in desc.supported_indexes:
            parts.append("")
            parts.extend(_render_index(idx))

    return "\n".join(parts)


def _render_index(idx: "IndexDescriptor") -> list[str]:
    lines: list[str] = []
    lines.append(f"  [{idx.name}]")
    lines.append(f"    {idx.description}")
    lines.append("")

    if idx.build_params:
        lines.append("    Build parameters:")
        for p in idx.build_params:
            lines.append("    " + _render_param(p))
    else:
        lines.append("    Build parameters: (none)")

    lines.append("")

    if idx.search_params:
        lines.append("    Search parameters:")
        for p in idx.search_params:
            lines.append("    " + _render_param(p))
    else:
        lines.append("    Search parameters: (none)")

    return lines


def _render_param(p) -> str:
    req = " (required)" if p.required else ""
    default = f"  [default: {p.default}]" if p.default is not None else ""
    return f"  --{p.name} <{p.type}>{req}{default}\n      {p.description}"
