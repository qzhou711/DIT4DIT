"""Model and dataset registry for plugin-style extensibility.

Allows registering custom backbones, decoders, and datasets by name so
that YAML configs can reference them without hardcoded imports.

Usage::

    from mimic_videos.core.registry import Registry

    @Registry.backbone("my_custom_backbone")
    class MyBackbone(nn.Module):
        ...

    # Instantiate by name later:
    backbone_cls = Registry.get_backbone("my_custom_backbone")
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class _Store:
    """Internal store for a category of registered objects."""

    def __init__(self, category: str) -> None:
        self._category = category
        self._table: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator factory — registers a class under *name*."""

        def _inner(cls: Type[T]) -> Type[T]:
            if name in self._table:
                raise KeyError(
                    f"[Registry] '{name}' is already registered in "
                    f"category '{self._category}'. Use a different name."
                )
            self._table[name] = cls
            return cls

        return _inner

    def get(self, name: str) -> type:
        if name not in self._table:
            available = sorted(self._table)
            raise KeyError(
                f"[Registry] '{name}' not found in category '{self._category}'. "
                f"Available: {available}"
            )
        return self._table[name]

    def list_registered(self) -> list[str]:
        return sorted(self._table)

    def __contains__(self, name: str) -> bool:
        return name in self._table

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Store(category={self._category!r}, keys={self.list_registered()})"


class Registry:
    """Central registry for all pluggable components.

    Categories:
        - ``backbone``   – video backbone models
        - ``decoder``    – action decoder models
        - ``dataset``    – robot episode datasets
        - ``schedule``   – flow-matching noise schedules
    """

    _backbones: _Store = _Store("backbone")
    _decoders: _Store = _Store("decoder")
    _datasets: _Store = _Store("dataset")
    _schedules: _Store = _Store("schedule")

    # ------------------------------------------------------------------ #
    # Decorator registration helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def backbone(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a backbone class under *name*."""
        return cls._backbones.register(name)

    @classmethod
    def decoder(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a decoder class under *name*."""
        return cls._decoders.register(name)

    @classmethod
    def dataset(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a dataset class under *name*."""
        return cls._datasets.register(name)

    @classmethod
    def schedule(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a noise schedule class under *name*."""
        return cls._schedules.register(name)

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def get_backbone(cls, name: str) -> type:
        return cls._backbones.get(name)

    @classmethod
    def get_decoder(cls, name: str) -> type:
        return cls._decoders.get(name)

    @classmethod
    def get_dataset(cls, name: str) -> type:
        return cls._datasets.get(name)

    @classmethod
    def get_schedule(cls, name: str) -> type:
        return cls._schedules.get(name)

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def list_backbones(cls) -> list[str]:
        return cls._backbones.list_registered()

    @classmethod
    def list_decoders(cls) -> list[str]:
        return cls._decoders.list_registered()

    @classmethod
    def list_datasets(cls) -> list[str]:
        return cls._datasets.list_registered()

    @classmethod
    def list_schedules(cls) -> list[str]:
        return cls._schedules.list_registered()
