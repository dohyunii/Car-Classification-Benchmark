# Preserve0.97551 and train only two missing I0 heads

Open `F0_I0_SixHead_Ensemble.ipynb` in Colab with a T4 GPU. It is self-contained and uses your previously reported Drive paths by default. Edit those only if you moved files.

Required saved artifacts:

- The exact F0three-head folder, including `ensemble_summary.json` and allthreehead`.pt`files.
- The exact I0ViT-B/14seed2026checkpoint from the scored0.97551blend.
- That winning cross-backbone run folder, including `summary.json` and `F0_I0_equal_branch_ensemble_submission.csv`.
- The original I0 feature cache under `student_outputs_I0/cache_vitb14/`. The notebook finds its exact signature filename automatically. It never silently substitutes/re-extracts different views.

Run all cells. Enable Colab Secret KAGGLE_API_TOKEN for the course download, or set DATA_ROOT and disable downloading. Cache metadata, dataset content hashes, labels, train-only normalization, sourcecheckpointSHA checks and original prediction reproduction gates must pass. There are no placeholder paths for the known successful runs, no head2026 retraining and no normalization refitting. If the cache is missing, restore it from Drive/the previous I0 run before proceeding.

Only new heads42and123 are trained. Both use3500fixed768-DP3trainingfeatures, augmentationseed123 and the original E4/AdamW recipe:768→128,BatchNorm,ReLU,Dropout.3,20classes;lr/decay.001,batch64,max35epochs,patience7. Every head and optimizer is initialized afresh and shuffling is independently seeded. Validation determines each head's first strictly best-accuracy epoch. No external data, pseudo-labels, test-time training, TTA or ensemble-weight search occurs.

The original F0threeheads and I0seed2026head are preserved byte-for-byte. New checkpoints use a separate `student_outputs_SixHead_Ensemble/<uuid>/`. Backbone inference uses the recorded pinned official source and actual weight fingerprints, runs sequentially for memory, and has a pinnedGit fallback for GitHub502/503/504 errors. Full SHA verification and TLS remain enabled.

Fixed final rule:

```
0.5 * mean(F0 seed42,123,2026 probabilities)
+ 0.5 * mean(I0 seed42,123,2026 probabilities)
```

Each of the six heads therefore contributes1/6. No head is discarded based on validation or Kaggle scores. Each branch uses its own saved training-derived normalization. The original four-head control must reproduce its source validation metrics and every label in the scored referenceCSV before any new candidate is exported.

Submit **`SixHead_equal_branch_ensemble_submission.csv`** manually as the sole new primary candidate. Keep the0.97551submission. The output also contains `CurrentBest_four_head_control.csv` (verified identical labels), `I0_three_seed_ensemble_control.csv`, bothnewI0checkpoints, traininghistories, validationprobabilities/comparison and provenanceJSON. No upload to Kaggle is performed, and no new Kaggle score is claimed.

Verification: notebook schema and code compilation, plus synthetic end-to-end tests of shared-cache integrity, separate384/768stats, twofreshheads, seed2026trainingrefusal, allsourcefilesunchanged, equalweights, originalCSVreproduction and testIDordering. Actual source models/features live in your Colab/Drive, not the development workspace, so real training and predictions must run there.
