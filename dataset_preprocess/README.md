# Evaluation dataset preprocessing

Run commands from the repository root after installing `requirements.txt`.
Each converter documents its expected raw directory structure at the beginning
of the Python file and exposes additional options through `--help`.

```bash
# 7Scenes
python dataset_preprocess/eval/seven_scenes.py \
  --input_dir /path/to/raw/7Scenes \
  --output_dir data/eval/7Scenes

# Neural RGB-D
python dataset_preprocess/eval/nrgbd.py \
  --input_dir /path/to/raw/NRGBD \
  --output_dir data/eval/NRGBD

# HiRoom
python dataset_preprocess/eval/hiroom.py \
  --input_dir /path/to/raw/HiRoom \
  --output_dir data/eval/HiRoom

# Synth4K (writes Synth4K-1 through Synth4K-5)
python dataset_preprocess/eval/synth4k.py \
  --input_dir /path/to/raw/InfiniDepth \
  --output_dir data/eval/Synth4K
```

Every output dataset contains a `.index.txt` and one directory per sample:

```text
data/eval/<Dataset>/
├── .index.txt
└── scene/frame/
    ├── image.jpg
    ├── depth.png
    └── meta.json
```

Set `--num_workers 1` when diagnosing malformed source data. The default uses
multiple workers where appropriate.
