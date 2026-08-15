# Evaluation

## 1. Prepare datasets

Download NYUv2, KITTI, ETH3D, iBims-1, Sintel, DDAD, DIODE, and HAMMER from
[Hugging Face](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/tree/main),
then extract them under `data/eval`.

The extracted directories should look like:

```text
data/eval/
├── NYUv2/
├── KITTI/
├── ETH3D/
├── iBims-1/
├── Sintel/
├── DDAD/
├── DIODE/
└── HAMMER/
```

Each dataset directory should contain its own `.index.txt`.

Preprocess 7Scenes, NRGBD, HiRoom, and Synth4K with the included converters:

```bash
python dataset_preprocess/eval/seven_scenes.py --input_dir /path/to/raw/7Scenes --output_dir data/eval/7Scenes
python dataset_preprocess/eval/nrgbd.py --input_dir /path/to/raw/NRGBD --output_dir data/eval/NRGBD
python dataset_preprocess/eval/hiroom.py --input_dir /path/to/raw/HiRoom --output_dir data/eval/HiRoom
python dataset_preprocess/eval/synth4k.py --input_dir /path/to/raw/InfiniDepth --output_dir data/eval/Synth4K
```

The exact raw directory layouts are documented at the beginning of each
converter. See [dataset_preprocess/README.md](../dataset_preprocess/README.md)
for the shared output format.

Update the `path` of every enabled dataset in:

```text
configs/eval/all_benchmarks.json
```

## 2. Run evaluation

```bash
python scripts/eval.py \
  --checkpoint checkpoints/pxdepth/model.pt \
  --config configs/eval/all_benchmarks.json \
  --output results/pxdepth.json \
  --fp16
```

The aggregated metrics are saved to `results/pxdepth.json`.

## Save predictions

Add `--dump-pred`:

```bash
python scripts/eval.py \
  --checkpoint checkpoints/pxdepth/model.pt \
  --config configs/eval/all_benchmarks.json \
  --output results/pxdepth.json \
  --fp16 \
  --dump-pred
```

Prediction files are written to `results/pxdepth_dump/`. Add `--dump-gt` if ground-truth visualizations are also needed.

## Other options

- `--fp32`: run in FP32 instead of FP16
- `--device cuda:1`: select another device
- `--input-size WIDTHxHEIGHT`: override the default `1022x770` input area
- `--fixed-size`: resize directly to `--input-size`
- `python scripts/eval.py --help`: show every option
