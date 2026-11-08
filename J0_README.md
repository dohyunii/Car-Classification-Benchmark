# J0: controlled accuracy improvement experiment

This compares frozen **DINOv2-S/14 at 224 versus 336 pixels**, keeping P3, E4, the original 700/200 split and all training settings fixed. It does not assume that a larger model or higher resolution is better. The predeclared experiment has two configurations and classifier seeds `[42, 123, 2026]`; augmentation seed is fixed at 123. The same pretrained parameters are used for both resolutions. No additional labeled images, original Stanford Cars labels, pseudo-labeling, or validation/test training is used.

## Run

Use Python with compatible PyTorch/TorchVision and `timm`, `scikit-learn`, `pandas`, `Pillow`, `matplotlib`. `accuracy_experiments.py` reuses dataset validation, image loading, seed utilities and hashes from `preprocessing_benchmark.py`; both files must be present. `J0_Resolution_Improvement.ipynb` embeds these needed utilities for self-contained Colab use. T4 GPU is recommended. A CPU run is possible but feature extraction is slow.

```bash
python accuracy_experiments.py --data-root /path/to/course/data --output-root outputs/J0 --batch-size 8
```

The dataset must have the course `train/`, `val/`, flat `test/`, and `sample_submission.csv`. Model downloads use the public timm DINOv2 Hugging Face repository; no API key is required. No automatic dataset download is performed. Provide the authorized course export. Outputs use a unique run directory. Existing F0/I0/H2/H3 files and artifacts remain untouched.

For Colab, open the new notebook, choose T4 GPU, set `DATA_ROOT` and `STORAGE_ROOT`, and run all. Mount Drive if desired. Features are stored on CPU; batch size 8 is the starting point. There is no Kaggle submission API call.

## Fixed settings

- Backbone: `vit_small_patch14_dinov2.lvd142m`, timm port of self-supervised DINOv2-S/14; frozen, evaluation mode, FP32, CLS token, 384-D output. Dynamic image sizing reuses one pretrained model at both resolutions.
- Preprocessing: 224/256 or 336/384 crop/resize sizes, bicubic; one deterministic original plus four random geometric training views; RandomResizedCrop scale .8–1, ratio .9–1.1, horizontal flip .5, no ColorJitter. Exactly 3,500 correlated training embeddings.
- Evaluation: deterministic single view; no TTA. Validation images are never trained on. Test extraction occurs only after resolution selection.
- Standardization: mean and population SD from all training embeddings only, SD clamped at 1e-6.
- E4: Linear(384,128), BatchNorm, ReLU, Dropout(.3), Linear(128,20). Each head and optimizer are new and seeded; shuffling has its own generator.
- AdamW, lr .001, weight decay .001, batch size 64, at most 35 epochs, patience 7. Strict best validation accuracy determines the checkpoint; F1 does not reset patience.
- Resolution is selected by mean validation accuracy across the three seeds, then mean macro-F1. The final head seed is **fixed to 2026**, not selected by its individual validation score. This rule was specified before seeing metrics.

Historical F0=98% and user-reported I0=97.5% are references, not measured by this script. The controlled 224px run is the primary comparison because this experiment uses the timm implementation rather than the original torch.hub implementation; interpolation details can differ. Mean/sample SD over three seeds reflects training randomness on one small validation set, not an independent estimate of generalization. Small differences should not be treated as certain improvements.

## Reproducibility and outputs

Cache identity records model/weights, size/transforms, augmentation seed, view layout, content-hashed ordered manifest, class mapping, extraction batch size and software/device. Head seed is excluded. Cache metadata, shapes, labels and finite values are checked. The current implementation includes TorchVision/Pillow versions and uses a shared cache directory for future invocations; the initial executed run used a per-run cache and recorded its software versions separately. These maintenance changes do not alter numerical defaults.

Outputs include per-run histories, per-image validation predictions, metrics and head checkpoints with train-only statistics and mappings; combined `J0_runs.csv`, `J0_summary.csv`, comparison PNG, `summary.json`, and a uniquely located local submission CSV. No old checkpoint is loaded as a new head. The test IDs retain original CSV string order and predictions are bounded integer labels. There is no automatic Kaggle upload.

Focused synthetic tests: `python -m unittest discover -s tests -p test_accuracy_experiments.py -v`. They do not claim real accuracy. The experiment results report, when produced, distinguishes actual runs from historical references and synthetic tests.
