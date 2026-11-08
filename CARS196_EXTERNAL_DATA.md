# Authorized Cars196 expansion: audit before training

Additional labeled Stanford Cars data was authorized in this conversation. The original competition train/validation/test folders remain unchanged. External source labels must not be used to recover competition test labels or produce ground-truth test predictions.

## Pinned public sources

The first attempted Kaggle mirror denied access. These public image/annotation mirrors were downloaded instead:

- `Multimodal-Fatima/StanfordCars_train`, revision `31847c49b43333c42e492bb31a21d3fcfcddab55`
- `Multimodal-Fatima/StanfordCars_test`, revision `44480308e3192efdfbb2ec4cec40b36feacbfbb7`

Both are on Hugging Face. Their source class ontologies must agree exactly. The six parquet shards contain 8,144 + 8,041 = 16,185 images. Repository revisions and local shard SHA-256 digests are recorded in `/workspace/shared/cars196/source_provenance.json`. Raw data is outside the Git checkout and is not committed.

## Processing order

1. `prepare_external_cars.py` reads only parquet image bytes, writes opaque source filenames and an image-to-shard-row index, and records source checksums. It does not interpret row-level labels.
2. `leakage_audit.py` compares all source images with all course training, validation and test images, independently of source labels. File SHA-256 and shape-aware decoded-RGB hashes catch exact copies; pHash/dHash over full, center90%, center75% and flipped variants flag related copies. Exact/near duplicate connected components propagate exclusions. Components touching validation/test are quarantined; training duplicates are redundant; only one representative survives a clean external component. Broad source/course ambiguous candidates are excluded, not assumed distinct.
3. `build_external_training_manifest.py` verifies allowlist/decision hashes, opaque image/index/shard joins and source revisions. It selects only allowed rows before interpreting their labels, maps the complete source class names to the 20 course classes exactly, and discards images of the other176 classes. Source labels for excluded images are not interpreted/exported, and source labels are never joined to competition test IDs.
4. Only the resulting verified positive manifest may be passed to the extra-data training program. No course validation/test image may enter model training or normalization fitting. The unchanged validation set remains the evaluation split. Kaggle uploading is manual.

The audit is conservative and heuristic: it is not a mathematical proof that every possible edit/crop has been detected. Ambiguous matches are excluded. A train-only calibration tested30 images×6 synthetic transforms (180 positives) and2,000 sampled different-class pairs: all180 positives detected, no sampled negative matches. This supports the tested transformations only; it does not establish zero population false positives/negatives. Calibration report is retained with the audit outputs.

## Reproduce

Install compatible PyTorch/TorchVision plus `numpy`, `scipy`, `Pillow`, `pandas`, `pyarrow`, `huggingface_hub`, `timm`, `scikit-learn` and `matplotlib`.

Use `huggingface_hub.snapshot_download` with `repo_type='dataset'`, the pinned revisions above and `allow_patterns=['data/*.parquet','README.md']`; preserve public API dataset metadata as `train_source.json`/`test_source.json` under the source root. Never disable TLS/checksum verification. Model and dataset downloads require their destination hosts to be allowed.

```bash
python prepare_external_cars.py /workspace/shared/cars196
python leakage_audit.py --external-image-root /workspace/shared/cars196/images --course-root /path/to/course --output-root outputs/Cars196_audit
python build_external_training_manifest.py --source-root /workspace/shared/cars196 --audit-dir <audit-run> --course-root /path/to/course --output-root outputs/Cars196_manifest
```

The source's public `test` split is distinct from the course's protected test split. Source images from either public partition are eligible only after overlap exclusion; its published labels are never a replacement for competition predictions.

Counts and model outcomes from actual execution are recorded in `CARS196_EXTERNAL_RESULTS.md` when available. Zero surviving target-class images is a valid audit outcome, and does not justify relabeling protected images or counting duplicate copies as new data.
