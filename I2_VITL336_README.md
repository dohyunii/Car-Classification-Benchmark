# I2: frozen DINOv2 ViT-L/14 at336px

Open `I2_ViTL336_Ensemble.ipynb` in Colab. Select T4 GPU; `LARGE_BATCH_SIZE=2` initially, or1 if memory is insufficient. A faster GPU is optional. Existing compatible PyTorch/TorchVision are retained. Model weights are substantially larger than ViT-B and extraction will take longer; do not interrupt a working pass solely because it is slower.

## Source artifacts

The known successful Drive paths are prefilled:

- F0 three-head run: `student_outputs_F0_Ensemble/20261108T091250Z_4ed04fc8/` with its summary and threeheadcheckpoints.
- Winning I1 B336 run: `student_outputs_I1_ViTB336/1402790cb5a0/` with `summary.json`, the B336 checkpoint and `F0_I1_336_equal_branch_submission.csv` scored0.97755.

Edit only if moved. No old I0cache or six-head experiment is required. Original SHA fingerprints, classmapping/data hashes, validation scores and every winningCSV prediction must match before L336extraction starts. Existing models are not retrained. Both sourcebackbones run sequentially and are freed before loading the new Lmodel.

Enable Colab Secret KAGGLE_API_TOKEN for the course download, or disable downloading and set DATA_ROOT. Run all cells. No helper uploads or manual initialization of fingerprint/parameter-count globals are required.

## Fixed candidate

New backbone: official `dinov2_vitl14` pretrained SSL weights, source SHA7764ea0f912e53c92e82eb78a2a1631e92725fc8, frozen/eval/inference mode, **1024-D CLS embeddings**. Its actual weight digest is recorded in metadata; there is no random-weight fallback. Transient GitHub502/503/504 errors use a pinnedGit checkout with revision verification and normal verified pretrained-weight loading. TLS/checksum verification is not disabled.

P3 at336: deterministic resize384/center336view plus four independently sampled random geometric views, scale.8–1,ratio.9–1.1,flip.5,BICUBIC,ImageNet normalization, augmentationseed123, noColorJitter. Sequential extraction/workers0. Expected training3500×1024, labels3500, validation200×1024 and testsamplelength×1024. These are correlated views of700original images.

Fit new per-feature mean/populationSD only on Ltraining features (clamp1e-6). B768statistics/features/headweights are not substituted. Fresh E4input1024→128,BatchNorm,ReLU,Dropout.3,20classes, headseed2026; AdamWlr/decay.001,batch64,max35epochs,patience7,firststrictlybestaccuracy. Headconfig and normalization are saved in a separate Lcheckpoint. No backbone fine-tuning, additional labeled data, pseudo-labeling, test-time learning or TTA occurs.

The sole primary candidate is fixed before evaluation:

```
(2/3) * current winning ensemble + (1/3) * L336
= (1/3) * F0 three-head mean + (1/3) * B336 + (1/3) * L336
```

The five classifierheadweights are1/9,1/9,1/9,1/3,1/3. No ensembleweights, seed subsets or alternative resolutions are searched. Combined-validation improvement is not required for export after source-integrity gates pass.

## Outputs and what to submit

A new `student_outputs_I2_ViTL336/<uuid>/` contains:

- **`F0_B336_L336_equal_backbone_submission.csv`**: manually submit this one primary candidate.
- `I2_ViTL336_standalone.csv`: separately labeled standalone diagnostic, not an automatically selected alternative.
- `CurrentBest_097755_control.csv`: verified equal to the original winning predictions.
- Lheadcheckpoint, traininghistory, validationcomparison/probabilities and provenanceJSON.

The shared `cache_336/` has a separate Larchitecture/resolution/content/seed/weights/software/batch-size key. Metadata, shape,dtype,finitevalues andlabels are validated; stale or corrupt caches regenerate. Explicit FORCE_REEXTRACT affects I2only. Original checkpoint hashes are rechecked before export. No Kaggleupload is performed. Preserve0.97755as the reference until a new score is measured; larger backbones can lose and a finite public leaderboard can be overfit.

## Validation

Synthetic end-to-end tests cover source replay, oldfilesunchanged,1024-D cache/head schema, repeatedlabels, train-onlystatistics, distinct768vs1024cache invalidation, cachehits, fixedweights, sampleordering/bounds, compiled/self-contained notebook imports and control CSVidentity. They use a tinyfakeencoder and do not measure DINOv2-L accuracy. Real pretrained downloads, T4memory/runtime, Ltraining and Kaggleperformance remain for the Colab execution. There are no fabricated Lresults in the notebook.
