# I3: conservative B336 fine-tuning

Open [I3_B336_FineTune_Ensemble.ipynb](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/I3_B336_FineTune_Ensemble.ipynb), select a T4 GPU and run all cells. The notebook is self-contained; no helper upload is required. Existing PyTorch/TorchVision are retained.

Known Drive run paths are prefilled: F0 `20261108T091250Z_4ed04fc8`, I1 `1402790cb5a0`, I2 `6dcb19f3d7e4`. Keep the entire source folders and the original B336/L336 feature caches under their `student_outputs_I*/cache_336/` directories. These caches are required: the notebook checks their exact source metadata, signature, labels, dimensions and finite values. It does not repeat L336 extraction. Missing caches stop with their required signatures. Enable the Colab secret `KAGGLE_API_TOKEN`, or disable downloading and supply the extracted course dataset.

Original checkpoint hashes and dataset content are checked. The original S224 heads are replayed; B336 and L336 probabilities are computed from their saved feature caches and heads. Every label of the current 0.97959 submission must reproduce before training. B336 image inference must also match its cached validation probabilities and source accuracy.

Only official DINOv2-B/14 blocks 10/11, final LayerNorm and the E4 head train. The first ten blocks remain frozen. The backbone retains evaluation behavior with autograd; head dropout trains, while BatchNorm running statistics stay fixed and its affine parameters train. Original training-derived feature mean/std remain fixed.

The fixed recipe is seed 2026, batch 4, backbone learning rate 2e-6, head learning rate 3e-5, AdamW decay 1e-4, gradient clipping 1, at most ten epochs, patience three. Only the original 700 training images receive fresh geometric augmentation: random resized 336px crop (scale .8–1, ratio .9–1.1), horizontal flip .5, ImageNet normalization. Evaluation remains resize384/center336. Validation selects the first strictly better accuracy; epoch zero is eligible and ties do not replace it. No validation/test images update parameters. The assignment lists DINOv2 as allowed and encourages fine-tuning; this is a separate accuracy experiment from the controlled frozen-backbone P0–P4 benchmark.

The sole candidate replaces the B336 branch in the fixed equal-backbone blend:

```
(F0 three-head probability mean + fine-tuned B336 + original L336) / 3
```

S and L weights/models stay fixed. There is no leaderboard weight search, extra data, pseudo-labeling or TTA. This may improve or reduce Kaggle accuracy; the small validation set cannot guarantee the result.

A new `student_outputs_I3_B336_FineTune/<uuid>/` holds configuration, training history, summary and `CurrentBest_097959_control.csv`. If a later epoch beats the original B336 validation accuracy, it also contains the full model `I3_B336_finetuned.pt` (including backbone, head and normalization buffers), saved probabilities, and **`F0_I3_B336_L336_equal_backbone_submission.csv`**. Manually submit that candidate and retain the whole output directory. If epoch zero wins, no new candidate is exported; retain the current 0.97959 submission. Original source files are never overwritten, and checkpoint hashes are checked again before completion. No Kaggle submission happens automatically.

Verification uses synthetic models and fixtures: frozen/trainable gradient boundaries, fixed BN statistics, best-epoch restoration, cache acceptance/rejection, fixed probability weights, source file preservation, ordered CSV/full-checkpoint export and notebook schema/self-contained definitions. These tests do not measure actual DINOv2 accuracy or GPU runtime. Real training runs in Colab.
