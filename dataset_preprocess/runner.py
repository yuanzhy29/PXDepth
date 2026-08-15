"""Parallel task execution and deterministic index generation.

This module is deliberately independent of dataset implementations. It keeps
process-pool, progress, error-summary, and index logic out of each adapter.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from tqdm import tqdm

from storage import Sample, write_index, write_sample


@dataclass(frozen=True)
class TaskResult:
    """Compact result sent from one worker back to the parent process.

    Args:
        entries: Relative sample keys successfully written by one task.
        stats: Counter-like values such as ``ok``, ``skip``, and ``error``.
        errors: Short diagnostics populated only when conversion fails.

    Returns:
        Picklable worker-result container consumed by :func:`run_dataset`.
    """

    entries: tuple[str, ...] = ()
    stats: Mapping[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


def add_common_args(parser: argparse.ArgumentParser, *, workers: int | None = None) -> argparse.ArgumentParser:
    """Add shared conversion arguments to one dataset CLI parser.

    Args:
        parser: Parser that may already contain dataset-specific options.
        workers: Default worker count. ``None`` selects half of available CPUs.

    Returns:
        The same parser, extended with input/output, worker, encoding, index,
        and strict-error options.
    """

    default_workers = workers if workers is not None else max(1, (os.cpu_count() or 8) // 2)
    parser.add_argument("--input_dir", required=True, help="Root of the extracted raw dataset.")
    parser.add_argument("--output_dir", required=True, help="Destination root in PXDepth processed format.")
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of conversion workers.")
    parser.add_argument("--index_name", default=".index.txt", help="Index filename written below output_dir.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality when RGB must be encoded.")
    parser.add_argument("--strict", action="store_true", help="Stop at the first malformed task.")
    return parser


def processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Extract :class:`BaseDataset` constructor values from parsed options.

    Args:
        args: Namespace containing options created by :func:`add_common_args`.

    Returns:
        Keyword dictionary with input/output paths, worker count, index name,
        JPEG quality, and strict-error behavior.
    """

    return {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "num_workers": args.num_workers,
        "index_name": args.index_name,
        "jpeg_quality": args.jpeg_quality,
        "strict": args.strict,
    }


def _as_samples(value: Sample | Iterable[Sample] | None) -> Iterator[Sample]:
    """Normalize one conversion result to a sample iterator.

    Args:
        value: One sample, an iterable of samples, or ``None`` for a skipped
            worker task.

    Returns:
        Iterator yielding zero or more :class:`Sample` objects.
    """

    if value is None:
        return iter(())
    if isinstance(value, Sample):
        return iter((value,))
    return iter(value)


def _execute(item: tuple[Any, Any]) -> TaskResult:
    """Convert and serialize one task inside a worker process.

    Args:
        item: Picklable ``(dataset, task)`` pair. The dataset supplies
            ``convert``, output options, and strict-error behavior.

    Returns:
        Successfully written keys and compact status/error information.
    """

    dataset, task = item
    try:
        entries = tuple(
            write_sample(dataset.output_dir, sample, jpeg_quality=dataset.jpeg_quality)
            for sample in _as_samples(dataset.convert(task))
        )
        return TaskResult(entries=entries, stats={"ok": len(entries)}) if entries else TaskResult(stats={"skip": 1})
    except Exception as exc:
        if dataset.strict:
            raise
        message = f"{task!r}: {type(exc).__name__}: {exc}"
        return TaskResult(stats={"error": 1}, errors=(message,))


def run_dataset(dataset: Any) -> list[str]:
    """Run discovery, conversion, serialization, and index generation.

    Args:
        dataset: Configured :class:`BaseDataset` instance. It must expose the
            runtime attributes and methods used in :func:`_execute`.

    Returns:
        Sorted duplicate-free keys written to the dataset index.
    """

    if not dataset.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {dataset.input_dir}")
    dataset.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(dataset.discover())
    if not tasks:
        raise RuntimeError(f"No {dataset.name} samples found under {dataset.input_dir}.")

    items = ((dataset, task) for task in tasks)
    executor: ProcessPoolExecutor | None = None
    if dataset.num_workers == 1:
        results = map(_execute, items)
    else:
        executor = ProcessPoolExecutor(max_workers=dataset.num_workers)
        results = executor.map(_execute, items, chunksize=1)

    entries: list[str] = []
    stats: Counter[str] = Counter()
    errors: list[str] = []
    try:
        for result in tqdm(results, total=len(tasks), desc=f"Preprocessing {dataset.name}", unit="task"):
            entries.extend(result.entries)
            stats.update(result.stats)
            errors.extend(result.errors[: max(0, 20 - len(errors))])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    entries = sorted(set(entries))
    index_path = write_index(dataset.output_dir, entries, dataset.index_name)
    print(f"Wrote {len(entries)} samples to {dataset.output_dir}")
    print(f"Wrote index: {index_path}")
    if stats:
        print("Stats: " + ", ".join(f"{key}={value}" for key, value in sorted(stats.items())))
    for error in errors:
        print(f"[WARN] {error}")
    if stats.get("error", 0) > len(errors):
        print(f"[WARN] {stats['error'] - len(errors)} additional errors omitted.")
    return entries
