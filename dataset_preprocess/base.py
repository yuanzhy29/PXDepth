"""Reusable parent class for dataset-specific RGB-D adapters.

Subclasses describe native frames through small read hooks. ``BaseDataset``
then performs common depth normalization, camera normalization, modality
validation, sample assembly, and delegates execution to ``runner.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

import cv2
import numpy as np

from depth import DepthData, prepare_depth
from runner import run_dataset
from storage import Sample, normalize_intrinsics, read_rgb as load_rgb


TaskT = TypeVar("TaskT")


class BaseDataset(ABC, Generic[TaskT]):
    """Base class for frame-based dataset conversion.

    Args:
        input_dir: Extracted raw dataset root.
        output_dir: Destination root in PXDepth format.
        num_workers: Worker count. One runs synchronously for debugging.
        index_name: Name of the generated sample index.
        jpeg_quality: JPEG quality for images that cannot be copied directly.
        strict: Re-raise malformed-task errors instead of recording them.

    Subclasses normally implement :meth:`discover`, :meth:`key`,
    :meth:`read_rgb`, :meth:`read_depth`, and :meth:`read_intrinsics`. The base
    class handles NaN/Inf conversion, shape checks, intrinsic normalization,
    sample assembly, multiprocessing, progress reporting, serialization, and
    deterministic index generation.

    A task should be lightweight and picklable. For video or HDF5 datasets,
    override :meth:`frames` so one worker opens a sequence once and yields many
    native frame descriptors. This avoids creating millions of futures while
    keeping every frame on the same common save path.
    """

    name = "dataset"
    resize_rgb_to_depth = False

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        num_workers: int = 1,
        index_name: str = ".index.txt",
        jpeg_quality: int = 95,
        strict: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.num_workers = max(1, int(num_workers))
        self.index_name = str(index_name)
        self.jpeg_quality = int(jpeg_quality)
        self.strict = bool(strict)

    @abstractmethod
    def discover(self) -> Sequence[TaskT] | Iterable[TaskT]:
        """Discover deterministic, picklable conversion tasks.

        Returns:
            Iterable of lightweight task objects consumed by :meth:`frames`.
        """

    def frames(self, task: TaskT) -> Iterable[Any]:
        """Expand one worker task into native frame descriptors.

        Args:
            task: Lightweight object returned by :meth:`discover`.

        Returns:
            Iterable of frame descriptors. By default, one task is one frame.
        """

        return (task,)

    @abstractmethod
    def key(self, frame: Any) -> str:
        """Return the relative output key for one frame.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            Relative POSIX path such as ``scene/camera/000001``.
        """

    @abstractmethod
    def read_rgb(self, frame: Any) -> np.ndarray | Path | str:
        """Read or locate the RGB image for one frame.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            RGB ``uint8 [H,W,3]`` array or a source image path. A JPEG path can
            be copied byte-for-byte by the common writer.
        """

    @abstractmethod
    def read_depth(self, frame: Any) -> DepthData | np.ndarray:
        """Read depth and describe source-specific invalid values.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            :class:`DepthData` or raw numeric ``[H,W]`` array. Datasets with
            explicit sky/far codes, unit scaling, or range limits should return
            ``DepthData`` so the base class applies them consistently.
        """

    @abstractmethod
    def read_intrinsics(self, frame: Any, width: int, height: int) -> np.ndarray:
        """Read pixel-space pinhole intrinsics for one output frame.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.
            width: Output depth width in pixels.
            height: Output depth height in pixels.

        Returns:
            Pixel-space ``float [3,3]`` camera matrix. The base class converts
            it to normalized coordinates exactly once.
        """

    def read_pose(self, frame: Any) -> np.ndarray | None:
        """Read an optional camera-to-world pose for one frame.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            ``float [4,4]`` camera-to-world matrix or ``None``.
        """

        return None

    def read_aux_depth(
        self,
        frame: Any,
        width: int,
        height: int,
        intrinsics: np.ndarray,
    ) -> Mapping[str, DepthData | np.ndarray]:
        """Read optional aligned depth inputs such as projected LiDAR.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.
            width: Output depth width in pixels.
            height: Output depth height in pixels.
            intrinsics: Pixel-space ``float [3,3]`` camera matrix returned by
                :meth:`read_intrinsics`.

        Returns:
            Mapping from output PNG filename to source ``[H,W]`` depth data.
        """

        return {}

    def read_segmentation(self, frame: Any) -> np.ndarray | None:
        """Read an optional integer segmentation map aligned to RGB/depth.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            Integer ``[H,W]`` label map or ``None``.
        """

        return None

    def segmentation_labels(self, frame: Any) -> Mapping[str, int] | None:
        """Return names for IDs produced by :meth:`read_segmentation`.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            Label-name to integer-ID mapping or ``None``.
        """

        return None

    def metadata(self, frame: Any) -> Mapping[str, Any]:
        """Return dataset-specific JSON metadata for one frame.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            JSON-serializable mapping. Intrinsics, resolution, and pose are
            added centrally and should not be duplicated here.
        """

        return {}

    def copy_jpeg(self, frame: Any, image: np.ndarray | Path | str) -> bool:
        """Choose whether a source JPEG is copied rather than re-encoded.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.
            image: RGB value returned by :meth:`read_rgb`.

        Returns:
            ``True`` for JPEG source paths by default, otherwise ``False``.
        """

        return isinstance(image, (str, Path)) and Path(image).suffix.lower() in {".jpg", ".jpeg"}

    def build_sample(self, frame: Any) -> Sample:
        """Apply common validation and assemble one processed sample.

        Args:
            frame: Native descriptor yielded by :meth:`frames`.

        Returns:
            :class:`Sample` with normalized intrinsics and canonical depth
            validity semantics, ready for the common serializer.
        """

        depth = prepare_depth(self.read_depth(frame))
        height, width = depth.shape
        image = self.read_rgb(frame)
        if self.resize_rgb_to_depth:
            array = load_rgb(image) if isinstance(image, (str, Path)) else np.asarray(image)
            if array.shape[:2] != (height, width):
                array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
            image = np.ascontiguousarray(array)

        intrinsics = self.read_intrinsics(frame, width, height)
        aux_depth = {
            name: prepare_depth(value)
            for name, value in self.read_aux_depth(frame, width, height, intrinsics).items()
        }
        return Sample(
            key=self.key(frame),
            image=image,
            depth=depth,
            intrinsics=normalize_intrinsics(intrinsics, width, height),
            meta=self.metadata(frame),
            pose=self.read_pose(frame),
            extra_depths=aux_depth,
            segmentation=self.read_segmentation(frame),
            segmentation_labels=self.segmentation_labels(frame),
            copy_jpeg=self.copy_jpeg(frame, image),
        )

    def convert(self, task: TaskT) -> Iterable[Sample]:
        """Convert every frame represented by one worker task.

        Args:
            task: One item returned by :meth:`discover`.

        Returns:
            Lazy iterable whose frames all pass through :meth:`build_sample`.
        """

        return (self.build_sample(frame) for frame in self.frames(task))

    def run(self) -> list[str]:
        """Run discovery, conversion, serialization, and index generation.

        Returns:
            Sorted keys written to the dataset index.
        """

        return run_dataset(self)
