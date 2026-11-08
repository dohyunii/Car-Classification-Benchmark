# J2: one bounded supervised fine-tuning experiment

J2 is the final predeclared candidate after J0's matched-resolution experiment and J1's train-only-selected RBF-SVM. It fine-tunes only the final two blocks of the same pretrained DINOv2-S/14 backbone and its E4 classifier. All updates use the provided 700 training images and their labels. No external labeled images, pseudo-labels, test training, test-driven selection or automatic Kaggle submission is used.

## Verified starting checkpoint

Use the selected J0 224px, augmentation-seed-123, fixed-head-seed-2026 checkpoint:

```text
outputs/J0/c8aab8360ba2/checkpoints/J0_S14_224_aug123_head2026.pt
```

This checkpoint measured **97.5% (195/200 correct)** in the current J0 experiment. That newly verified baseline is J2's starting point. It is not the historical user-reported F0 98% checkpoint. J2 does not bypass or change H2's separate historical-baseline gate.

Before optimization, J2 validates the source cache signature, class mapping, ordered content-hashed dataset manifest, 224px preprocessing metadata, view layout, source architecture and fixed seeds. Fresh pretrained parameters must exactly match the source J0 weight fingerprint; the E4 state is loaded strictly. The saved 384-D training-derived means and population standard deviations are retained as fixed buffers. J2 independently evaluates the restored model on all 200 validation images and must reproduce **195 correct and exactly 0.975 accuracy**. A mismatch raises an error before an optimizer is created; no random head or normalization fallback is allowed.

## Exact training recipe

- Backbone: `vit_small_patch14_dinov2.lvd142m`, the same pretrained timm DINOv2-S/14 used by J0.
- Trainable backbone: blocks **10 and 11**, plus final LayerNorm. All earlier blocks, patch embedding, positional/class tokens and other parameters remain frozen.
- E4 head: `Linear(384,128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128,20)`, restored from J0. Head parameters remain trainable.
- Keep the backbone in evaluation mode **with autograd enabled**. Training inputs and earlier frozen parameters require no gradients, so the earlier blocks do not retain a backward graph.
- During training the head uses active Dropout. BatchNorm running means, running variances and batch counts stay fixed; its affine weight/bias remain trainable. Evaluation puts the complete head in eval mode.
- Image size 224. Fresh P3 geometric augmentation per original training image per epoch: RandomResizedCrop(scale .8–1, ratio .9–1.1, bicubic), horizontal flip .5 and ImageNet normalization. This is image fine-tuning, not five offline feature copies per epoch. No ColorJitter or TTA.
- Validation uses deterministic Resize(256, bicubic)/CenterCrop(224) and ImageNet normalization. Validation/test images never fit statistics or update weights.
- Seed 2026; seeded shuffled training loader, zero workers, batch size 8.
- AdamW backbone LR **2e-6**, head LR **3e-5**, weight decay **1e-4**, global gradient clipping at **1.0**.
- Maximum **10 epochs**, patience **3**. There is no hyperparameter grid, extra scheduler or adaptive second candidate.

Epoch zero is evaluated and saved as the initial best checkpoint before updates. Retain a fine-tuned epoch **only for strictly higher validation accuracy**. An accuracy tie, even with higher macro-F1, does not replace the best or reset patience. After three non-improving epochs stop, restore the best full state and independently re-evaluate it. If fine-tuning never improves accuracy, retain epoch zero. Macro-F1 is reported as a secondary diagnostic.

## Execute

From the repository using the same compatible PyTorch/TorchVision/timm environment as J0:

```bash
python finetune_accuracy_experiment.py \
  --source-checkpoint outputs/J0/c8aab8360ba2/checkpoints/J0_S14_224_aug123_head2026.pt \
  --data-root /path/to/the/extracted/course/dataset \
  --output-root outputs/J2 \
  --threads 4
```

Dependencies are PyTorch, TorchVision, timm, NumPy, pandas, Pillow, scikit-learn and matplotlib, together with the repository helpers `accuracy_experiments.py` and `preprocessing_benchmark.py`. The source weights need to remain in the model cache or be downloadable through the configured proxy. No external API key is needed when data is already extracted.

For a diagnostic smoke test only, `--epochs 1` lowers the cap. The production default is 10 and larger values are rejected. Do not treat a short smoke test as the completed experiment. The implementation supports CUDA when available and falls back to CPU. **CUDA is unavailable in the current Codex environment; the actual CPU run was launched and was still pending when this guide was written.** No J2 validation improvement is claimed here. GPU compatibility has not been measured by that CPU run.

Each invocation creates a new UUID directory and preserves earlier J0/J1/F0/I0 artifacts. Progress includes the epoch, batches 1/41/81, training loss and complete per-epoch validation metrics. Stop/restart only deliberately; an interrupted run can already contain a valid epoch-zero or later best checkpoint, but it is not evidence that the ten-epoch/patience-capped workflow completed.

## Artifacts and interpretation

| File | Contents |
|---|---|
| `J2_config.json` | Exact settings, software/device versions, source checkpoint path/hash and pretrained weight fingerprint, train/eval behavior and selection rule. |
| `J2_best.pt` | Restorable full backbone/head state, fixed feature statistics, class mapping, source metadata, baseline/best metrics, epoch, config and frozen/trainable parameter counts. Epoch zero is a valid saved best. |
| `J2_history.csv` | Epoch zero plus every completed fine-tuning epoch, training loss, validation accuracy, macro-F1 and correct count. |
| `J2_validation_predictions.csv` | Restored best model's per-image validation path, true/predicted label and confidence. |
| `J2_summary.json` | Baseline/best metrics, selected epoch, improvement flag, epoch/runtime counts, checkpoint path and no-test/no-upload flags. Written after restoration succeeds. |

The training program intentionally performs **no test inference**. A separate prediction exporter is planned after checkpoint selection; do not invent a test submission path or Kaggle score. Any later exporter must use the selected restored full model and its fixed source statistics, preserve original sample ID order, and write a local CSV without uploading it.

Compare the final selected accuracy with the verified 97.5% epoch-zero baseline. A tie keeps the original E4 model. One additional correct validation image is 0.5 percentage points, so any small gain is provisional and is not proof of a Kaggle leaderboard improvement.

## Lightweight tests

```bash
python -m unittest discover -s tests -p test_finetune_accuracy_experiment.py -v
```

Synthetic tiny-backbone tests cover the exact trainable-mask, lack of earlier-block autograd graphs, nonzero trainable-block/final-norm/BN-affine gradients, unchanged frozen weights and BN running statistics under actual optimizer updates, changed trainable block weights, source/epoch-zero gates, optimizer refusal after a failed gate, strict accuracy-only patience, eligible/restored epoch zero, state/statistics/class mapping/source-metadata round-tripping and overwrite protection. These tests do not measure real DINOv2 accuracy.


## Executed result

Actual CPU run independently reproduced97.5% epoch0, then measured96.5%,97.5%,96.0% in epochs1–3. It stopped/restored epoch0. No J2 test predictions generated. export_finetuned_submission.py is now implemented but correctly refuses this non-improving checkpoint. See `ACCURACY_IMPROVEMENT_RESULTS.md` for the full comparison.
