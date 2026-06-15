from __future__ import annotations

import atexit
import json
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .util import (
    read_bool_env,
    read_str_env,
    read_int_env,
)

import torch


_MAX_PENDING_RECORDS = 16
_WRITER_GROUP_RANK = 0


@dataclass(frozen=True)
class LoadProfileConfig:
    enabled: bool
    output_dir: Path
    flush_interval: int
    record_interval: int


def load_profile_config() -> LoadProfileConfig:
    flush_interval = read_int_env("ULTRA_EP_LOAD_PROFILE_FLUSH_INTERVAL", 128)
    record_interval = read_int_env("ULTRA_EP_LOAD_PROFILE_RECORD_INTERVAL", 1)
    if flush_interval <= 0:
        raise ValueError("ULTRA_EP_LOAD_PROFILE_FLUSH_INTERVAL must be positive")
    if record_interval <= 0:
        raise ValueError("ULTRA_EP_LOAD_PROFILE_RECORD_INTERVAL must be positive")
    return LoadProfileConfig(
        enabled=read_bool_env("ULTRA_EP_LOAD_PROFILING", False),
        output_dir=Path(read_str_env("ULTRA_EP_LOAD_PROFILE_DIR", "ultra_ep_traces")),
        flush_interval=flush_interval,
        record_interval=record_interval,
    )


@dataclass
class _StagedPreLoad:
    layer_id: int
    real_layer_id: int
    microbatch: int
    pre_cpu: torch.Tensor
    pre_event: torch.cuda.Event


@dataclass
class _PendingRecord:
    layer_id: int
    real_layer_id: int
    microbatch: int
    pre_cpu: torch.Tensor
    post_cpu: torch.Tensor
    events: tuple[torch.cuda.Event, torch.cuda.Event]


class ExpertLoadProfiler:
    def __init__(
        self,
        config: LoadProfileConfig,
        group_rank: int,
        global_rank: int,
        metadata: dict[str, Any],
    ):
        self.config = config
        self.enabled = config.enabled and group_rank == _WRITER_GROUP_RANK
        self._closed = False
        self._staged: dict[int, deque[_StagedPreLoad]] = defaultdict(deque)
        self._pending: list[_PendingRecord] = []
        self._orphaned: list[_StagedPreLoad] = []
        self._records: list[dict[str, Any]] = []
        self._futures: list[Future] = []
        self._layer_microbatches: dict[int, int] = defaultdict(int)
        self._dropped = 0
        self._chunk_id = 0

        if not self.enabled:
            self._executor = None
            self.metadata = metadata
            self.trace_id = ""
            return

        timestamp = int(time.time() * 1000)
        self.trace_id = (
            f"epg{metadata['ep_group_id']}_gr{group_rank}_r{global_rank}_{timestamp}"
        )
        self.metadata = {
            **metadata,
            "trace_id": self.trace_id,
            "schema": "ultra_ep_trace_v1",
            "created_unix": time.time(),
            "writer_group_rank": _WRITER_GROUP_RANK,
            "max_pending_records": _MAX_PENDING_RECORDS,
            "record_interval": config.record_interval,
            "flush_interval": config.flush_interval,
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ultra_ep_load_profile"
        )
        self._write_manifest()
        atexit.register(self.close)

    def has_staged(self, layer_id: int) -> bool:
        return self.enabled and bool(self._staged.get(layer_id))

    def stage_pre(
        self,
        layer_id: int,
        real_layer_id: int,
        pre_logical_loads: torch.Tensor,
    ) -> None:
        if not self._can_record():
            return

        microbatch = self._layer_microbatches[real_layer_id]
        self._layer_microbatches[real_layer_id] = microbatch + 1
        if microbatch % self.config.record_interval != 0:
            return

        self._drain_ready()
        pre_cpu, pre_event = self._copy_to_pinned(pre_logical_loads)
        self._staged[layer_id].append(
            _StagedPreLoad(
                layer_id=layer_id,
                real_layer_id=real_layer_id,
                microbatch=microbatch,
                pre_cpu=pre_cpu,
                pre_event=pre_event,
            )
        )

    def record_post(self, layer_id: int, post_physical_loads: torch.Tensor) -> None:
        if not self._can_record():
            return
        staged_queue = self._staged.get(layer_id)
        if not staged_queue:
            return

        self._drain_ready()
        if len(self._pending) >= _MAX_PENDING_RECORDS:
            self._dropped += 1
            self._orphaned.append(staged_queue.popleft())
            if not staged_queue:
                self._staged.pop(layer_id, None)
            return

        staged = staged_queue.popleft()
        if not staged_queue:
            self._staged.pop(layer_id, None)

        post_cpu, post_event = self._copy_to_pinned(post_physical_loads)
        self._pending.append(
            _PendingRecord(
                layer_id=staged.layer_id,
                real_layer_id=staged.real_layer_id,
                microbatch=staged.microbatch,
                pre_cpu=staged.pre_cpu,
                post_cpu=post_cpu,
                events=(staged.pre_event, post_event),
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return

        for record in self._pending:
            for event in record.events:
                event.synchronize()
            self._records.append(self._materialize(record))
        self._pending.clear()
        for staged in self._orphaned:
            staged.pre_event.synchronize()
        self._orphaned.clear()
        for staged_queue in self._staged.values():
            for staged in staged_queue:
                staged.pre_event.synchronize()
        self._staged.clear()
        self._flush_records(force=True)

        for future in self._futures:
            future.result()
        self._futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._write_manifest()

    def _can_record(self) -> bool:
        if not self.enabled or self._closed:
            return False
        checker = getattr(torch.cuda, "is_current_stream_capturing", None)
        if checker is None:
            return True
        try:
            return not checker()
        except RuntimeError:
            return False

    @staticmethod
    def _copy_to_pinned(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
        src = tensor.detach()
        cpu = torch.empty(
            tuple(src.shape),
            dtype=src.dtype,
            device="cpu",
            pin_memory=True,
        )
        cpu.copy_(src, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(src.device))
        return cpu, event

    def _drain_orphaned(self) -> None:
        if not self._orphaned:
            return
        self._orphaned = [
            staged for staged in self._orphaned if not staged.pre_event.query()
        ]

    def _drain_ready(self) -> None:
        self._drain_orphaned()
        if not self._pending:
            self._check_futures()
            return

        remaining = []
        for record in self._pending:
            if all(event.query() for event in record.events):
                self._records.append(self._materialize(record))
            else:
                remaining.append(record)
        self._pending = remaining
        self._flush_records()
        self._check_futures()

    @staticmethod
    def _materialize(record: _PendingRecord) -> dict[str, Any]:
        return {
            "layer": record.real_layer_id,
            "virtual_layer": record.layer_id,
            "microbatch": record.microbatch,
            "pre": record.pre_cpu.numpy().copy(),
            "post": record.post_cpu.numpy().copy(),
        }

    def _flush_records(self, force: bool = False) -> None:
        if not self._records:
            return
        if not force and len(self._records) < self.config.flush_interval:
            return
        records = self._records
        self._records = []
        chunk_id = self._chunk_id
        self._chunk_id += 1
        assert self._executor is not None
        self._futures.append(
            self._executor.submit(
                _write_chunk,
                self.config.output_dir,
                self.trace_id,
                chunk_id,
                self.metadata,
                self._dropped,
                records,
            )
        )

    def _check_futures(self) -> None:
        if not self._futures:
            return
        unfinished = []
        for future in self._futures:
            if future.done():
                future.result()
            else:
                unfinished.append(future)
        self._futures = unfinished

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        manifest = {
            **self.metadata,
            "chunks": self._chunk_id,
            "dropped_records": self._dropped,
        }
        path = self.config.output_dir / f"{self.trace_id}.manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp.replace(path)


def _write_chunk(
    output_dir: Path,
    trace_id: str,
    chunk_id: int,
    metadata: dict[str, Any],
    dropped: int,
    records: list[dict[str, Any]],
) -> None:
    import numpy as np

    layers = np.asarray([record["layer"] for record in records], dtype=np.int32)
    virtual_layers = np.asarray(
        [record["virtual_layer"] for record in records], dtype=np.int32
    )
    microbatches = np.asarray(
        [record["microbatch"] for record in records], dtype=np.int64
    )
    pre = np.stack([record["pre"] for record in records]).astype(np.int32, copy=False)
    post = np.stack([record["post"] for record in records]).astype(np.int32, copy=False)
    chunk_metadata = {
        **metadata,
        "chunk_id": chunk_id,
        "records": int(layers.size),
        "dropped_records": dropped,
    }

    path = output_dir / f"{trace_id}.chunk{chunk_id:06d}.npz"
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            metadata=np.asarray(json.dumps(chunk_metadata, separators=(",", ":"))),
            layers=layers,
            virtual_layers=virtual_layers,
            microbatches=microbatches,
            pre_logical_loads=pre,
            post_physical_loads=post,
        )
    tmp.replace(path)
