# Self-contained F0 notebook with flip TTA

Open `F0_DINOv2_TTA.ipynb` in Google Colab and select **Runtime → Change runtime type → T4 GPU**. No helper uploads are required.

## Fresh run: no old checkpoint needed

Leave `MODE='train_f0'`. Enable Colab Secret `KAGGLE_API_TOKEN` with notebook access and accept the competition rules; `DOWNLOAD_FROM_KAGGLE=True` downloads the course dataset and sets DATA_ROOT automatically. Alternatively set it False and provide extracted `train/`, `val/`, flat `test/` and original `sample_submission.csv`.

Run all cells. The notebook loads official frozen DINOv2-S/14, extracts cached P3 features (one original plus four geometric views, seed123), fits training-only standardization and trains a fresh E4 head with seed2026. E4 is Linear384→128, BatchNorm, ReLU, Dropout.3, Linear128→20. AdamW lr/decay.001, batch64, max35epochs, patience7; strictly best validation accuracy selects the restored head.

Fresh training is not recovery of the exact previous F0 weights. The historical98% validation and0.94693Kaggle scores are user-reported references, not promised outputs. Official upstream model code is pinned to SHA7764ea0f912e53c92e82eb78a2a1631e92725fc8. Source and pretrained weights require network access in Colab; no random-weight fallback is used.

## Reuse the exact original F0

Set `MODE='load_checkpoint'`, upload/mount the original `dinov2_p3_e4_aug123_head2026.pt` and set `F0_CHECKPOINT` accordingly. Keep `EXPECTED_LOADED_ACCURACY=0.98` for that verified model. The notebook validates architecture, seed metadata, class mapping and feature statistics, skips extraction/training and checks that the starting score reproduces before test prediction. A missing/incompatible checkpoint or mismatch stops execution; diagnose it rather than substituting a J0/timm model or bypassing the gate.

## TTA and files to use

Inference uses the same head/statistics for the deterministic center crop and its horizontal flip, averaging the two **probability** vectors equally. No head/backbone updates, normalization fitting, pseudo-labeling, additional source images or TTA-weight tuning occur. Validation/test are never used for classifier training.

Every execution creates a new `student_outputs_F0_TTA/<timestamp_uuid>/`; the safe content/weight/config-aware cache is shared in its separate `cache/` directory. `USE_DRIVE=True` persists these under MyDrive. Prior F0/I0/H2 checkpoints and submissions are untouched. Rerunning the same training cell within one run should be avoided; rerun configuration to create a new run directory. CSV writing refuses overwrite.

The final cell prints a nested `F0_flip_TTA_<uuid>/` containing:

- `F0_original_comparison.csv`: the original inference candidate.
- `F0_flip_TTA_submission.csv`: the requested flip-TTA candidate.
- `summary.json`: original/TTA validation metrics, paired changes, checkpoint/source identity and flags confirming no submission.
- `validation_comparison.csv`: measured accuracy, macro-F1 and correct/count.

Both CSVs preserve sample ID string ordering, contain integer labels0..19, and have exactly the sample length (700 for the course export). Compare the two files manually under Kaggle submission limits. The notebook does **not** upload either. A small validation loss does not establish a test loss or gain; one validation image changes accuracy by0.5percentage points.

## Verification

Notebook schema validation and code compilation passed. A sequential synthetic integration test executed both train and checkpoint modes, verified 3500×384 feature/label alignment, train-only normalization, strict384-D E4 restoration, shared-cache metadata, starting-score reproduction, paired TTA evaluation and identical ordered CSVs across train/restore modes. The four helper tests also cover probability averaging, labels ignored during inference, eval behavior, paired statistics, overwrite protection and the portable F0 gate.

Actual pretrained F0 training, Kaggle/Drive authentication and T4 execution were not run in this development environment. No synthetic scores are stored as benchmark results.
