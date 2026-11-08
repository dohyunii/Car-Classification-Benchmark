# Fixed horizontal-flip TTA

For a self-contained Colab notebook, use [F0_DINOv2_TTA.ipynb](F0_DINOv2_TTA.ipynb). It can train a fresh F0 or restore the original checkpoint; see [F0_TTA_README.md](F0_TTA_README.md).

A small validation split gives uncertain comparisons. On200 images, one correct prediction changes accuracy by0.5 percentage points. A small validation loss does not prove TTA will lose on Kaggle, and it also does not establish a Kaggle gain. Compare one fixed candidate; avoid selecting TTA weights/crops or repeatedly searching against the leaderboard.

This experiment averages **probabilities** from the deterministic original center crop and its horizontal flip, with equal weights. It uses a frozen backbone, head eval mode (no Dropout/BatchNorm updates), and the original saved training-only feature statistics. No training, label changes, test-time learning, or automatic Kaggle submission occurs.

## Executed local J0 comparison

`flip_tta_experiment.py` runs on the existing J0 checkpoint, not the unavailable original F0 checkpoint. Results and paired predictions are saved in its unique `outputs/Flip_TTA/<run>/` directory. Both original and TTA test CSVs are exported regardless of a small validation delta, as requested. Use the original sample submission ID order. Compare these two J0 candidates against one another to isolate TTA; comparing J0TTA to an oldF0 leaderboard score mixes model and inference changes.

```bash
python flip_tta_experiment.py --checkpoint outputs/J0/c8aab8360ba2/checkpoints/J0_S14_224_aug123_head2026.pt --data-root /path/to/course
```

## Apply to the actual F0 model in Colab

The original F0 checkpoint/cache are still unavailable locally. `F0_flip_tta_colab_cell.py` contains standalone functions and no repository-module imports. Paste/run that file's contents **inside the existing F0 notebook after restoring/evaluating the exact model that produced98%**. Keep deterministic `val_ds` and `test_ds` transforms. The verified original model, its own head and its own feature statistics must be retained; do not substitute the J0 backbone or train a new head in this step.

Then call:

```python
tta_output = run_verified_f0_flip_tta(
    backbone, head, feature_mean, feature_std,
    val_ds, test_ds, required_ids, OUTPUT_DIR, DEVICE,
    expected_accuracy=0.98,
)
```

The gate requires the original98% validation score to reproduce before test prediction. It creates a new UUID directory with `F0_original_comparison.csv`, `F0_flip_TTA_submission.csv`, and measured summary JSON. It does not submit them. If your verified starting model has another measured accuracy, supply that actual expected accuracy rather than bypassing reproduction failures. The generic prediction helper also supports I0's768-D features when paired with that model's own head/statistics.

Report actual Kaggle scores only after manually submitting under the competition's submission limits. A public leaderboard is another finite sample; any apparent gain should remain provisional until final evaluation.
