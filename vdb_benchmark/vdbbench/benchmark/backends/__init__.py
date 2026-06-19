"""Backend registry -- auto-discovers backend packages at import time.

Every sub-directory of ``backends/`` that contains an ``__init__.py``
with a module-level ``backend_descriptor`` attribute (a callable
returning :class:`BackendDescriptor`) is loaded and registered
automatically.

Public API consumed by the rest of the benchmark:

*  ``registry`` -- the singleton :class:`BackendRegistry`.
*  ``get_backend(name)`` -- shortcut to instantiate a backend by name.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Dict, List, Optional, Type

from .base import (
    BackendDescriptor,
    CollectionInfo,
    IndexDescriptor,
    IndexProgress,
    ParamDescriptor,
    VectorDBBackend,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Data model
    "BackendDescriptor",
    "CollectionInfo",
    "IndexDescriptor",
    "IndexProgress",
    "ParamDescriptor",
    "VectorDBBackend",
    # Registry
    "BackendRegistry",
    "registry",
    "get_backend",
]


class BackendRegistry:
    """Collects :class:`BackendDescriptor` instances from backend packages.

    Only **active** backends (``descriptor.active is True``) are visible
    through the public query methods (``get``, ``names``,
    ``list_backends``, ``create_backend``).  Inactive backends are still
    stored internally so they can be reactivated at runtime if needed.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, BackendDescriptor] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, descriptor: BackendDescriptor) -> None:
        """Register a backend descriptor (idempotent for the same name)."""
        key = descriptor.name.lower()
        if key in self._backends:
            logger.debug("Backend '%s' already registered; skipping.", key)
            return
        self._backends[key] = descriptor
        status = "active" if descriptor.active else "inactive"
        logger.debug("Registered backend: %s (%s)", key, status)

    # ------------------------------------------------------------------
    # Querying  (only active backends)
    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[BackendDescriptor]:
        """Return the descriptor for *name*, or ``None``.

        Returns ``None`` for inactive backends.
        """
        desc = self._backends.get(name.lower())
        if desc is not None and not desc.active:
            return None
        return desc

    def list_backends(self) -> List[BackendDescriptor]:
        """Return all **active** registered descriptors, sorted by name."""
        return sorted(
            (d for d in self._backends.values() if d.active),
            key=lambda d: d.name,
        )

    def names(self) -> List[str]:
        """Return **active** registered backend names, sorted."""
        return sorted(k for k, d in self._backends.items() if d.active)

    def __contains__(self, name: str) -> bool:
        desc = self._backends.get(name.lower())
        return desc is not None and desc.active

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def create_backend(self, name: str) -> VectorDBBackend:
        """Instantiate and return a (disconnected) backend by name.

        Raises :class:`ValueError` for unknown or inactive backends.
        """
        desc = self.get(name)
        if desc is None:
            available = ", ".join(self.names()) or "(none)"
            raise ValueError(
                f"Unknown backend '{name}'. Available: {available}"
            )
        return desc.backend_class()

    # ------------------------------------------------------------------
    # Introspection (includes inactive)
    # ------------------------------------------------------------------
    def all_backends(self, include_inactive: bool = True) -> List[BackendDescriptor]:
        """Return every registered descriptor, optionally including inactive ones."""
        return sorted(
            (d for d in self._backends.values() if include_inactive or d.active),
            key=lambda d: d.name,
        )


# Singleton used by the rest of the package.
registry = BackendRegistry()


def get_backend(name: str) -> VectorDBBackend:
    """Convenience: instantiate a backend by name from the global registry."""
    return registry.create_backend(name)


# ------------------------------------------------------------------
# Auto-discovery
# ------------------------------------------------------------------

def _discover_backends() -> None:
    """Walk sub-packages of this directory and register any that expose
    a ``backend_descriptor`` callable.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for finder, subpkg_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if not is_pkg:
            continue  # skip plain .py files like base.py
        fqn = f"{__name__}.{subpkg_name}"
        try:
            mod = importlib.import_module(fqn)
        except Exception:
            logger.warning(
                "Failed to import backend package '%s'; skipping.",
                fqn, exc_info=True,
            )
            continue

        descriptor_fn = getattr(mod, "backend_descriptor", None)
        if descriptor_fn is None:
            logger.debug(
                "Package '%s' has no backend_descriptor(); skipping.", fqn
            )
            continue

        try:
            desc = descriptor_fn() if callable(descriptor_fn) else descriptor_fn
            if isinstance(desc, BackendDescriptor):
                registry.register(desc)
            else:
                logger.warning(
                    "backend_descriptor in '%s' did not return a "
                    "BackendDescriptor; got %s",
                    fqn, type(desc).__name__,
                )
        except Exception:
            logger.warning(
                "Error calling backend_descriptor() in '%s'; skipping.",
                fqn, exc_info=True,
            )


_discover_backends()
