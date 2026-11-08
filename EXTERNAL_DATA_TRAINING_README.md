The external-data experiment uses only target-class Cars196 images retained by
the completed image-only overlap audit and approved training manifest. Source
labels are interpreted by `build_external_training_manifest.py` after exclusion;
`external_data_training.py` never parses source annotations. It rechecks the
approved manifest SHA256, audit files, source provenance, image index, class
mapping, source cards, verified source shards, image hashes and allowlist
membership before extracting anything. It rejects course paths, exact course
copies, duplicate additions and components containing any course image.

Run after the audit and manifest builder finish:

```bash
python external_data_training.py \
  --data-root /path/to/course-dataset \
  --source-root /path/to/cars196 \
  --audit-dir outputs/Cars196_audit/AUDIT_ID \
  --manifest outputs/Cars196_manifest/MANIFEST_ID/external_training_manifest.json \
  --manifest-sha256 MANIFEST_FILE_SHA256 \
  --base-cache outputs/J0/c8aab8360ba2/cache/J0_S14_224_ae02839d767a7c0104c9e33286f90d463083180c427dbfe71b5496df67e47540.pt \
  --baseline-run outputs/J0/c8aab8360ba2
```

The original 3,500 raw course training features and 200 validation features are
reused unchanged. Only approved new images pass through the frozen
`vit_small_patch14_dinov2.lvd142m` encoder, whose actual weight SHA256 must match
the original cache. Each receives P3's original view and four geometric views at
224 pixels, augmentation seed 123, and zero data-loader workers. Five-view
external features are appended after the unchanged course features. Feature mean
and population standard deviation use combined training rows only.

Fresh E4 heads use seeds 42, 123 and 2026, 384→128→20 dimensions, BatchNorm,
ReLU, dropout 0.3, AdamW learning rate/weight decay 0.001/0.001, batches of 64,
35 maximum epochs and patience 7. The first strictly best validation accuracy
selects each checkpoint, exactly as J0. If the last shuffled batch contains one
row, it joins the preceding batch, producing a final batch of 65; every training
row still appears exactly once per epoch. This exception is recorded per run.

Unique run directories contain provenance, combined feature cache, train-only
normalization statistics, checkpoints, validation predictions, histories,
three-seed metrics, sample-standard-deviation summaries and matched per-seed
deltas against the verified J0 course-only control. Original caches and runs are
never overwritten. An empty eligible manifest produces an explicit skipped
summary and no feature extraction or training. The command never runs course
test inference or creates/uploads a Kaggle submission. The overlap audit remains
heuristic; its limitations are preserved in the run provenance.

Focused synthetic checks (a tiny fake encoder, no pretrained download or course
training):

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_external_data_training.py'
```
