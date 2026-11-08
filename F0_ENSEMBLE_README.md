# F0 three-seed ensemble

Use `F0_ThreeSeed_Ensemble.ipynb` in Colab with a T4 GPU. It is self-contained and uses the same official DINOv2-S/14, P3/E4 and training settings as `F0_DINOv2_TTA.ipynb`. Backbone and training-derived feature statistics are fixed; fresh classifier heads use seeds42,123,2026. All three softmax probability vectors are averaged equally. No seed/weight/subset search, TTA, additional labeled data or automatic Kaggle submission is performed.

1. Upload/open the notebook in Colab; select T4 GPU.
2. Enable Colab Secret KAGGLE_API_TOKEN, or disable downloading and set DATA_ROOT to extracted course data.
3. To reuse features, retain the previous compatible `student_outputs_F0_TTA/cache/` under STORAGE_ROOT. Set USE_DRIVE=True to share persistent Drive storage across notebooks/runtimes, or copy that cache folder to the new runtime. The notebook checks cache identity; missing or incompatible features regenerate once, not per seed.
4. Run all. Each execution creates a new `student_outputs_F0_Ensemble/<timestamp_uuid>/` with all three checkpoints, histories, individual validation scores, equal-ensemble metrics and four CSVs.
5. Manually submit **F0_three_seed_ensemble_submission.csv** as the new candidate. Keep **F0_seed2026_submission.csv** as a same-run control. Historical0.95510 is a reference, not guaranteed by new training. No validation-improvement gate blocks export.

The original F0/TTA notebooks and artifacts remain unchanged. Training uses only the original700 labeled images (3500P3 views), normalization uses those training features only, validation remains200images, and test IDs follow the original sampleCSV. Classifier settings stay AdamWlr/decay.001, batch64, max35epochs, accuracy-only checkpoint selection and patience7. There is no test-time training or pseudo-labeling.

Schema validation, code compilation and a synthetic sequential integration test passed. The test verifies one backbone load/one feature cache, three independent fresh heads and seeded checkpoints, shared statistics, exact probability averaging, ID ordering and no-upload flags. It does not measure real F0/ensemble accuracy or Kaggle performance. Real training and submission remain for Colab; retain the run directory before ending the runtime.
