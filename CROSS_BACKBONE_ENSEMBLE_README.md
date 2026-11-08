# Fixed F0 + I0 cross-backbone ensemble

Next candidate after the user-reported **0.95918** F0 three-seed Kaggle result. This is not a measured cross-backbone score. Use `F0_I0_CrossBackbone_Ensemble.ipynb` in Colab with a T4 GPU; it is self-contained and does not require repository helper uploads.

## Inputs: preserve the exact source models

Copy/mount both saved model outputs, ideally in Drive:

1. The F0 run directory that generated the0.95918submission, containing `ensemble_summary.json` and **all three `.pt` checkpoints** for seeds42,123,2026. The notebook resolves their saved basenames inside that folder, so moving the whole folder to Drive is supported.
2. The I0 ViT-B/14 seed2026 checkpoint, typically `I0_dinov2_vitb14_p3_e4_aug123_head2026.pt`, from the earlier97.5%validation run.

CSV labels alone are insufficient: blending requires probability predictions from the original models. If a checkpoint was lost, rerunning the corresponding source notebook produces a new model, not guaranteed identical to the scored one. Do not silently replace it and attribute the score to the original.

## Colab steps

1. Open the notebook and select **T4 GPU**.
2. Set `F0_RUN_DIR` and `I0_CHECKPOINT` in section1. Keep `USE_DRIVE=True` for Drive paths, or disable it for uploaded files under `/content`.
3. Enable Colab Secret `KAGGLE_API_TOKEN` and notebook access for the course download. Alternatively set `DOWNLOAD_FROM_KAGGLE=False` and edit `DATA_ROOT` to the extracted course export.
4. Run all. Original source scores must reproduce before the branch's test inference; class mappings, source weight hashes, E4 architecture, seeds, statistics and I0's dataset content manifest must match. A failure is diagnostic, not permission to bypass the check.
5. Manually submit **`F0_I0_equal_branch_ensemble_submission.csv`** as the new candidate. Keep the existing0.95918submission. The generated `F0_three_seed_control.csv` is its matched control; `I0_ViTB14_control.csv` is also available.

## Exact rule

```
p_F0 = (p_S42 + p_S123 + p_S2026) / 3
p_candidate = 0.5 * p_F0 + 0.5 * p_B2026
```

Thus the four effective head weights are1/6,1/6,1/6,1/2, **not**1/4each. Every classifier uses its own saved training-only normalization statistics. Probability columns share the exact20-class mapping; rows share original sample-submission ID order. There is no retraining, fitting statistics on validation/test, TTA, learned weights, seed-subset selection, pseudo-labels or additional labeled data in this step.

The same pinned official DINOv2 implementation used for the sources is loaded, with pretrained hashes checked. ViT-S and ViT-B inference runs sequentially, each in frozen eval/inference mode. Small-model embeddings are computed once per batch and shared across the three S heads. Batch8 is the default; CPU inference is possible but slower.

## Outputs and interpretation

A new `student_outputs_F0_I0_Ensemble/<uuid>/` contains the three candidate/control CSVs, measured validation comparison, validation probability arrays and `summary.json` with source-checkpoint hashes and fixed weights. No previous model/output is overwritten. No Kaggle upload is performed.

The blend is exported regardless of a small validation delta after source reproduction checks pass. It might help or hurt Kaggle; current performance remains unknown. Do not repeatedly optimize weights against a finite public leaderboard.

Notebook schema/code compilation and synthetic integration tests passed: one forward pass serves all S heads; distinct384/768stats and heads restore; all source reproduction gates are enforced; branch weights, CSV ordering, bounds, no-overwrite and no-training/no-upload flags are checked. Real checkpoints live in the user's Colab outputs and were not available in the development workspace, so no real cross-backbone metrics are claimed.
