"""Measure tensor storage retained by immutable inference state."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RetainedTensorMemory:
    resident_bytes: int
    unique_storages: int
    tensor_references: int


def retained_tensor_memory(value: Any) -> RetainedTensorMemory:
    """Sum unique underlying tensor storages reachable from ``value``.

    Aliases and views are counted once by device, storage pointer, and storage
    size. Storage bytes, rather than Python-object size or peak allocator usage,
    describe the reusable state that must remain resident between renders.
    """

    storages: dict[tuple[str, int, int], int] = {}
    visited_objects: set[int] = set()
    tensor_references = 0

    def visit(item: Any) -> None:
        nonlocal tensor_references
        if isinstance(item, torch.Tensor):
            tensor_references += 1
            storage = item.untyped_storage()
            storage_bytes = storage.nbytes()
            key = (str(item.device), storage.data_ptr(), storage_bytes)
            storages[key] = storage_bytes
            return
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            return
        identity = id(item)
        if identity in visited_objects:
            return
        visited_objects.add(identity)
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                visit(child)

    visit(value)
    return RetainedTensorMemory(
        resident_bytes=sum(storages.values()),
        unique_storages=len(storages),
        tensor_references=tensor_references,
    )
