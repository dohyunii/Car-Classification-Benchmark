# I5: three seeds per S/B336/L336 backbone

[Open I5 in Colab](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/I5_ThreeSeed_Backbone_Ensemble.ipynb). Select **Runtime → Change runtime type → T4 GPU**, enable notebook access to the Colab secret `KAGGLE_API_TOKEN`, then **Runtime → Run all**. Authorize Drive. The successful original F0/I1/I2 paths are prefilled; leave them unchanged unless moved. This does not use I3 fine-tuned weights or I4 flipped views.

The original winning ensemble has three S224 heads but only one B336 and one L336 head. I5 preserves those five checkpoints and adds B336 seeds42/123 and L336 seeds42/123, using the original saved frozen features. The two seed2026 checkpoints are never retrained. Average seeds42/123/2026 within each backbone, then average the S/B/L branches. All nine heads therefore have weight1/9. There is no seed-subset or ensemble-weight search.

New heads use E4: input768 or1024 → Linear128 → BatchNorm → ReLU → Dropout.3 → Linear20. Each trains for a **fixed three epochs**, matching the training duration of the saved B/L seed2026 models, with AdamW lr.001, weightdecay.001, batch64 and an independently seeded shuffle generator. Validation is diagnostic only: no early stopping, best-epoch selection or export veto based on validation improvement. The old checkpoints remain their original selected epoch3; the four new heads use their final epoch3. No model/backbone updates outside those four heads occur.

Keep the original B336/L336 feature caches in `student_outputs_I*/cache_336/`. They contain3500 training rows (five correlated views of700 original training images),200 validation rows and700 test rows. Full-cache metadata/signature, dimensions, finite values, repeated training labels, validation labels and test order are checked. Their recomputed training mean/population SD must match the saved source statistics; normalization stays fixed and uses no validation/test fitting. The actual dataset must match the original content manifest.

Only S224 image inference is replayed to verify the original three heads. B/L original predictions and all four new heads use saved features: there is **no B336/L336 model download or feature extraction**. Source checkpoint hashes, classmapping, original validation scores, and every label in the original0.97959 submission must reproduce before head training starts. Head training should be substantially lighter than backbone extraction; actual Colab runtime depends on the session.

Every run creates `student_outputs_I5_NineHead/<uuid>/` containing:

- **`I5_S_B336_L336_three_seed_submission.csv`** — the single new Kaggle candidate.
- `CurrentBest_097959_control.csv` — verified original predictions.
- Nine head checkpoints: byte-identical copies of the five original files and four new files.
- Per-new-head training histories, `summary.json`, `test_prediction_changes.csv`, and `probabilities.npz` with original/candidate outputs and all three B/L head probabilities.

Download the candidate from the printed Drive path and submit it manually to the course competition. Suggested description: `I5: equal S224 B336 L336, three head seeds per backbone, no TTA`. Send back the score and the printed changed-test-prediction count. Keep the whole output folder and the original0.97959 submission. More seeds can lower accuracy; no improvement is guaranteed. Nothing is uploaded to Kaggle automatically.

Synthetic tests verify exact three-epoch execution despite worsening validation, train-only normalization checks, view-major labels, preserved source weights, saved-head probability replay, equal nine-head averaging, full pipeline export despite worse validation, nine saved head files, and notebook schema/self-contained definitions. Real DINOv2/Kaggle accuracy is measured only by the user's Colab run and submission.
