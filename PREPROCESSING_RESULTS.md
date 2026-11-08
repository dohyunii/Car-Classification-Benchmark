# P1–P4 preprocessing results

Executed on the provided car dataset using pretrained frozen DINOv1 ViT-S/16 and a linear 384→20 classifier on CPU. Global, augmentation and head seeds were all 123. Split sizes: 700 training, 200 validation, 700 unlabeled test images. No Kaggle leaderboard score was measured.

| Experiment | Validation accuracy | Macro-F1 | Correct / 200 | Best epoch | Epochs run | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 77.5% | 0.7659 | 155 | 34 | 35 | 93.6 s |
| P2 | 78.5% | 0.7788 | 157 | 34 | 35 | 97.5 s |
| P3 | 84.0% | 0.8377 | 168 | 22 | 29 | 258.2 s |
| P4 | 83.5% | 0.8311 | 167 | 21 | 28 | 267.1 s |

P1: single geometric view. P2: single geometric + ColorJitter view. P3: deterministic view plus four geometric views. P4: deterministic view plus four geometric + ColorJitter views.

P3 had the highest validation accuracy in this run. P3 and P4 differ by just one correct prediction (0.5 percentage points). This is one seed only; seed variability is unmeasured. Repeat P3 and P4 on seeds 42, 123 and 2026 before selecting a winner. P0 was subsequently run with seed 123; see the P0–P4 comparison below.

All runs used AdamW, cross-entropy, learning rate 0.001, weight decay 0.001, maximum 35 epochs, patience 7 and head batch size 64. Multi-view experiments contain 3,500 correlated embeddings from 700 images and more optimizer updates per epoch. Validation/test transforms were deterministic and standardization used training views only.

Artifacts: `outputs/metrics/p1_p4_seed123.csv`, per-run JSON/history/confusion CSVs, `outputs/checkpoints/`, `outputs/cache/`, and comparison/history PNGs in `outputs/plots/`. Generated artifacts are ignored by Git.

## Multi-seed P3/P4 comparison

Completed augmentation/head seeds [42, 123, 2026]; global seed remained 123. The completed seed-123 runs were reused. All six checkpoints use identical frozen pretrained backbone weights.

| Experiment | Augmentation/head seed | Validation accuracy | Macro-F1 | Best epoch |
|---|---:|---:|---:|---:|
| P3 | 42 | 83.5% | 0.8312 | 24 |
| P3 | 123 | 84.0% | 0.8377 | 22 |
| P3 | 2026 | 83.0% | 0.8273 | 9 |
| P4 | 42 | 83.5% | 0.8320 | 24 |
| P4 | 123 | 83.5% | 0.8311 | 21 |
| P4 | 2026 | 83.5% | 0.8328 | 9 |

| Experiment | Accuracy mean ± sample SD | Macro-F1 mean ± sample SD |
|---|---:|---:|
| P3 | 83.50% ± 0.50 pp | 0.8320 ± 0.0053 |
| P4 | 83.50% ± 0.00 pp | 0.8319 ± 0.0009 |

P3 and P4 tie on mean validation accuracy (83.5%). P4 has lower observed seed variation for both accuracy and macro-F1. P3 has only a negligible mean macro-F1 advantage (~0.0001). P4 is the more consistent configuration in these three runs, but three seeds on the same 200-image validation set do not establish general superiority. Zero observed accuracy SD means all three runs got 167/200 correct, not that their predictions or future performance are identical. These are validation results, not Kaggle leaderboard scores.

Exports: `outputs/metrics/p3_p4_multiseed_runs.csv` and `outputs/metrics/p3_p4_multiseed_summary.csv`; exact-run comparison and seed-summary plots are in `outputs/plots/`. Standard deviations use ddof=1.

## P0–P4 seed-123 comparison

P0 subsequently completed with all seeds set to 123, using the same split, frozen pretrained DINOv1 and head training settings.

| Experiment | Validation accuracy | Macro-F1 | Best epoch |
|---|---:|---:|---:|
| P0 | 81.0% | 0.8079 | 31 |
| P1 | 77.5% | 0.7659 | 34 |
| P2 | 78.5% | 0.7788 | 34 |
| P3 | 84.0% | 0.8377 | 22 |
| P4 | 83.5% | 0.8311 | 21 |

P0 classified 162/200 validation images correctly. Runtime: 109.2 seconds on CPU. P3 improves on P0 by 3.0 percentage points (six validation images) for seed 123; P4 improves by 2.5 points (five images). P1 and P2 are below P0 for this seed. P0 has not been evaluated across multiple seeds. These remain validation comparisons, not Kaggle leaderboard results.

CSV: `outputs/metrics/p0_p4_seed123.csv`.
