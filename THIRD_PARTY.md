# Third-party components

PXDepth incorporates and depends on open-source projects that remain subject to their respective licenses.

## Included source

- **DINOv2**, Meta Platforms, Inc. The minimal ViT backbone implementation under `pxdepth/model/dinov2` is derived from DINOv2 and distributed under the Apache License 2.0. The repository-level [`LICENSE`](LICENSE) contains the applicable Apache License text.
- **MoGe**, Microsoft Corporation. Data utilities, geometric alignment utilities, and parts of the evaluation infrastructure are derived from MoGe and distributed under the MIT License. Its retained license is in [`LICENSES/MoGe.txt`](LICENSES/MoGe.txt).

Copyright notices in vendored source files are retained.

## Runtime dependencies

- `utils3d`: geometry utilities from EasternJournalist
- `pipeline`: asynchronous data pipeline from EasternJournalist
- PyTorch, TorchVision, NumPy, SciPy, OpenCV, Pillow, Matplotlib, Einops, Open3D, OpenEXR, HDF5/h5py, and Hugging Face Hub

Installations of these packages are governed by their own package licenses. This file is attribution information and is not a replacement for those licenses.
