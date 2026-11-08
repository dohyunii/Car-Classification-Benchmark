# I1: official frozen ViT-B/14 at336px

Use `I1_ViTB336_Ensemble.ipynb` in Colab with a T4 GPU. It is self-contained; no helper uploads are required. The known Drive paths for your winning0.97551four-head run are prefilled. Edit them only if you moved the source files. The0.97346six-head candidate is not used.

## Run

1. Mount Drive and keep the exact F0three-head folder, I0seed2026checkpoint, and winning cross-backbone output folder (`summary.json` plus `F0_I0_equal_branch_ensemble_submission.csv`).
2. Enable Colab Secret KAGGLE_API_TOKEN and notebook access for the course dataset download, or disable downloading and set DATA_ROOT. Source paths are checked before any model work.
3. Run all. Batch8 is the default for frozen FP32 inference. Reduce it to4 if necessary; source reproduction gates must still pass. Frozen S and B backbones run sequentially. Source and pretrained weights use the recorded official upstream SHA; a pinnedGit fallback handles transient GitHub502/503/504 errors without changing the revision or disabling verification.
4. The notebook verifies original checkpoint hashes, class mappings, dataset file hashes, validation scores and every label of the winning CSV. A failure stops before the new336experiment; do not bypass it.
5. It then extracts **new336px features** and trains **one fresh head, seed2026**. No existing model is overwritten or retrained. A new UUID output folder is created, with a separate shared336cache.

Primary new CSV: **`F0_I1_336_equal_branch_submission.csv`**. Submit it manually as one new experiment; preserve the0.97551submission. `I1_ViTB336_standalone.csv` is a separately labeled standalone diagnostic. `CurrentBest_097551_control.csv` must match the original scored predictions. There is no automatic Kaggle upload or claim that either new CSV improves the score.

## Controlled change

- Same official frozen DINOv2-B/14 pretrained weights,768-D CLS features, P3 augmentation and E4 classifier recipe as I0. No timm backend substitution.
- Only resolution changes: crop224→336 and deterministic resize256→384, BICUBIC. Random crop scale.8–1, ratio.9–1.1, flip.5, ImageNet normalization; no ColorJitter/TTA.
- One deterministic original plus four random training views per image, augmentation seed123 and sequential extraction order/workers0. Exactly3500training embeddings from700distinct training images. Validation200images/testsamplelength each use one deterministic view.
- Recompute mean/populationSD on **new training features only**, clamp1e-6. Never apply old224statistics or old classifier weights to336features.
- Fresh E4:768→128, BatchNorm, ReLU, Dropout.3,20classes. Seed2026; AdamWlr/decay.001,batch64,max35epochs,patience7; first strictly best accuracy checkpoint, no F1tie-reset.
- Fixed new blend:50%unchanged F0three-head probabilitymean +50%new336Bhead. Effective headweights1/6,1/6,1/6,1/2. No weights, subsets or resolution search.
- Validation metrics are reported; new candidates export regardless of a small validation delta after source-integrity gates pass. No external labeled data, pseudo-labeling or validation/test training.

## Cache and artifacts

`student_outputs_I1_ViTB336/cache_336/` uses resolution, resize/transforms, actual weights/source, augmentation seed/viewlayout, classmapping, content-hashed source manifest/testIDs, extractionbatchsize, software anddevice in its signature. Shape,dtype,finitevalues,labels andmetadata are checked. Invalid caches regenerate; FORCE_REEXTRACT explicitly refreshes I1 only.224px caches cannot be reused.

Every run stores a unique `I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt` with head weights, new normalization, source/feature metadata, mapping, resolution, seeds and selected epoch. It also writes traininghistory, validationprobabilities/comparison, the threeCSVfiles and provenance summary. Original source-checkpoint hashes are checked again before export.

Notebook schema/compilation and synthetic integration tests passed: original controls/checkpoints are preserved; both224and336inputpaths execute; all5views/labels align; new train-only normalization differs appropriately from old stats; compatible cache reuse and corrupt-cache regeneration work; fixedweights and CSVordering/bounds hold. Actual pretrained336GPU extraction/training/Kaggle performance is not measured in development; it runs in your Colab. Higher resolution can lose, and the finite public leaderboard is not a replacement for an independent evaluation.
