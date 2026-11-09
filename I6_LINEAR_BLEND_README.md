# I6: linear classifiers alongside I5's MLP heads

[Open I6 in Colab](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/I6_Linear_MLP_Ensemble.ipynb). Select **Runtime → Change runtime type → T4 GPU**, enable notebook access to the Colab secret `KAGGLE_API_TOKEN`, and choose **Runtime → Run all**. Authorize Drive. The source path is prefilled as `MyDrive/Car-Classification-Benchmark/student_outputs_I5_NineHead/5e401a22b20f/`, the run reported to score0.98163. Keep its nine checkpoints, `summary.json`, `probabilities.npz` and winning CSV. Original B336/L336 caches must remain under `student_outputs_I*/cache_336/`.

This is a classifier-diversity experiment. Six new `Linear(feature_dim,20)` classifiers use B336's768-D and L336's1024-D frozen embeddings, seeds42/123/2026 each. Each trains for a **fixed20epochs**, AdamW lr.001, weightdecay.001, batch64. Each uses the same saved3500 training-feature rows (five correlated views of700 images) and source training-only mean/std. No validation/test statistics are fitted. Validation is diagnostic: it never selects the epoch, seed subset, blend weight, or whether to export. No additional data, TTA, backbone updates, backbone downloads or image inference occur.

The sole predeclared candidate is:

```
S = unchanged I5 S224 three-MLP mean
B = 0.75 * I5 B336 three-MLP mean + 0.25 * B336 three-linear mean
L = 0.75 * I5 L336 three-MLP mean + 0.25 * L336 three-linear mean
candidate = (S + B + L) / 3
```

All averaging uses class probabilities. Effective weights per head are1/9 for each of the three SMLPs,1/12 for each of the six B/LMLPs and1/36 for each of the six B/Llinear heads. This preserves equal backbone weights and favors the winning classifiers within each modified branch. No other mixtures are exported as alternative submissions.

Before training, source checkpoint hashes, class mapping, dataset content, original feature metadata/signatures, finite shapes, view-major labels and training normalization are verified. All six original I5 B/L heads replay from their caches and must match the archived per-head probabilities and validation counts. The Sbranch mean is recovered as `3*I5mean - Bmean - Lmean` from I5's saved probabilities. FP32 bounds, normalization and reconstruction are checked; only rounding-scale negative values are clamped and renormalized. I5's original CSV labels must match both the saved and reconstructed ensemble. Its source files are hashed again before export. No pretrained backbone is needed for this replay.

A new `student_outputs_I6_LinearBlend/<uuid>/` contains:

- **`I6_S_B336_L336_MLP_linear_blend_submission.csv`** — manually submit this single candidate.
- `CurrentBest_098163_control.csv` — the verified winning I5 predictions.
- Fifteen head checkpoint files: byte-identical copies of the nine I5 files plus six new linear heads.
- Six training histories, `summary.json`, `test_prediction_changes.csv` and `probabilities.npz` with original/candidate and branch probabilities.

The final cell prints the exact candidate path and changed-prediction count. Download the CSV from Drive or Colab's Files panel. In the course competition choose **Submit Prediction**, and use a description such as `I6: equal S/B336/L336, 75% MLP + 25% linear in B/L, no TTA`. Send back the score and changed-test-prediction count. Retain the complete output folder and keep0.98163 as the reference until a better score is measured. No Kaggle upload happens automatically; a different classifier can lower accuracy.

Targeted verification uses synthetic data/models to check fixed20epoch training despite worsening validation, linear checkpoint/probability replay, unchanged source statistics/weights, FP32 branch recovery, fixed mixture weights, strict source-hash rejection, full pipeline export despite worse validation, fifteen saved heads, sample ordering and self-contained notebook/schema checks. Actual accuracy is measured only by Colab execution and Kaggle submission.
