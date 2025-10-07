# Iris Segmentation (LightIrisNet)

A tidy project structure for training and testing an iris+pupil segmentation model.  
This repo includes thin entrypoints (`scripts/train.py`, `scripts/test.py`) and a package (`src/irisseg/`).  
Your commonly used **training** and **testing** flags are baked in as **defaults**, so you can run with no arguments.

---

## Quickstart

### 1) Create environment & install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Prepare data
Expected layout (customize with flags if your layout differs):
```
data/
├─ images/               # input images (used at test-time; train may read via dataset class)
├─ labels_iris/          # iris masks (optional at test-time)
├─ labels_pupil/         # pupil masks (optional at test-time)
├─ train.txt             # list of IDs used for training
├─ val.txt               # list of IDs used for validation
└─ test.txt              # list of IDs used for testing
```
- Text files contain **one ID per line** (without extension).  
- Paths are relative to `--root` (default is `.\data`).

### 3) Train (uses your defaults)
```bash
python -m scripts.train
```
Defaults (override any of these on the command line):
```
--root .\data
--train_split .\data\train.txt
--val_split .\data\val.txt
--img_size -1
--batch_size 8
--epochs 120
--lr 3e-4
--seed 42
--workers 8
--outdir runs\train
--paper_aug
--amp
--freeze_enc_bn
--use_ellipse
--backbone mobilenetv3
--aug_limbus
--extra_decoder_conv
--pupil_refine_depth 1
--iris_tversky
--pupil_tversky
--iris_boundary_boost 1.2
--pupil_boundary_boost 1.2
--iris_sdt_r 20
--pupil_sdt_r 16
--warmup_epochs 5
--grad_clip 1.0
--uncertainty_weighting
--iris_floor 1.1
--dataset_loss_scale
```

### 4) Test / Inference (uses your defaults)
```bash
python -m scripts.test
```
Key defaults:
```
--root .\data
--test_split .\data\test.txt
--checkpoint runs\train\best_mobilenetv3.pt
--backbone mobilenetv3
--use_ellipse
--extra_decoder_conv
--pupil_refine_depth 1
--amp
--save_overlay
--export_dir runs\irisparsenet_aug_all\miche_diag
--thr_iris 0.5
--thr_pupil 0.5
--containment_mode soft
--inside_thresh 0.85
--keep_area_frac 0.90
```
Other relevant defaults already present in the code:
```
--images_dir images
```
(Modify if your test images live elsewhere under `--root`.)

---

## Override examples

- Change dataset location:
```bash
python -m scripts.train --root D:\datasets\iris --train_split D:\datasets\iris\splits\train.txt --val_split D:\datasets\iris\splits\val.txt
```

- Switch backbone and turn off paper augmentations:
```bash
python -m scripts.train --backbone resnet50 --paper_aug False
```

- Test with a different checkpoint and output directory:
```bash
python -m scripts.test --checkpoint runs\train\best_resnet50.pt --export_dir runs\preds_resnet50
```

- Run with explicit image directory at test-time (if not `images/`):
```bash
python -m scripts.test --images_dir eval_images
```

---

## Repo layout

```
.
├─ scripts/
│  ├─ train.py   # thin CLI entrypoint → calls irisseg._train_impl.main()
│  └─ test.py    # thin CLI entrypoint → calls irisseg._test_impl.main()
├─ src/
│  └─ irisseg/
│     ├─ _train_impl.py  # original training implementation (defaults injected)
│     ├─ _test_impl.py   # original testing implementation (defaults injected)
│     ├─ data.py         # façade modules for cleaner imports (Phase 2: move logic here)
│     ├─ models.py
│     ├─ losses.py
│     ├─ metrics.py
│     ├─ geometry.py
│     ├─ engine.py
│     └─ utils.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> The façade modules re-export the original functionality so you can gradually refactor without breaking callers.

---

## Reproducibility tips

- The training script sets seeds and controls CUDA/CuDNN behavior (see code).  
- Keep `--seed` fixed for comparable runs.
- For CPU-only environments, leave `--amp` off. For GPUs, `--amp` is enabled by default here.

---

## Checkpoints & large files

- Checkpoints are saved under `runs/` (default `runs\train`).  
- If you push to GitHub, either:
  - **Don’t commit weights**: keep `*.pt` ignored (recommended), or
  - **Use Git LFS** to version `*.pt`/`*.pth`.

---

## Troubleshooting

- **Out-of-memory (OOM)**: lower `--batch_size`, set a smaller `--img_size`, or disable `--amp` if your GPU/driver mix is problematic.
- **Paths on macOS/Linux**: use `/` instead of `\`. Example: `--root ./data`.
- **Dataloader workers on Windows**: if you hit spawn issues, try `--workers 0`.
- **Mismatched checkpoint/backbone**: ensure `--backbone` matches the model used for training.

---

## How to install as a package (optional)

Editable install to allow `from irisseg import ...` anywhere:
```bash
pip install -e .
# then
python -c "import irisseg; print('ok')"
```

---

## License
Add a license file (MIT/Apache-2.0) if you plan to share publicly.
