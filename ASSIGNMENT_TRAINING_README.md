# Complete training notebook for the assignment

Use **[Final_Car_Classifier_Training.ipynb in Colab](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/Final_Car_Classifier_Training.ipynb)** for a readable end-to-end execution record. The earlier `Final_Car_Classifier.ipynb` and verified0.98376 artifact remain unchanged as inference backups.

This notebook starts from the provided images. It downloads the three official pretrained DINOv2 backbones, extracts all features afresh, initializes/trains all fifteen classification heads and combines their probabilities. It never loads a previous student model, head checkpoint, ensemble bundle, prediction archive or feature cache. The official pretrained SSL backbones are the only loaded model weights, as required by the SSL downstream method. Transformer layers use Facebook Research's pinned official implementation; both classifier architectures and their training loops are shown directly in the notebook.

## Run and preserve actual evidence

1. Open a fresh copy using the link above. Choose **Runtime → Change runtime type → T4 GPU**.
2. Enable notebook access to the Colab secret `KAGGLE_API_TOKEN`. Choose **Runtime → Run all** and authorize Drive. No previous experiment paths or checkpoints are needed.
3. Allow the full run to finish. L336 extraction is expensive and can take hours on T4. Its default batch size is2; S uses16, B uses8. The notebook prints elapsed time and remaining estimates. Changing batches/hardware can change numerical results.
4. Inspect the live losses, all15 head records, final validation metrics and training curves. The last cells print the actual output directory and `submission.csv`.
5. **Save/download the executed `.ipynb` with its outputs visible**, using File → Save a copy in Drive or File → Download → Download .ipynb. Code alone and an unexecuted notebook do not establish that the actual course run completed. No fake outputs or historical training results are prefilled.
6. Keep the entire `student_outputs_Assignment_Training/<run-id>/` folder. Submit its CSV manually to Kaggle, record the newly measured score and retain the screenshot/report required by the assignment. No Kaggle upload occurs automatically.

## Readable structure

The notebook has named sections for dependencies, fixed settings, dataset integrity, transformations, official frozen backbones, classifier definitions and training. Then three explicit stages extract/train S, B and L separately, followed by the probability ensemble and actual execution evidence. It is self-contained: no repository/helper uploads are required. `assignment_training.py` mirrors the readable functions for testing; the notebook includes them directly.

| Backbone | Input / features | Fresh MLP heads | Fresh linear heads |
|---|---|---|---|
| DINOv2 ViT-S/14 | 224px /384-D | seeds42/123/2026 | none |
| DINOv2 ViT-B/14 | 336px /768-D | seeds42/123/2026 | seeds42/123/2026 |
| DINOv2 ViT-L/14 | 336px /1024-D | seeds42/123/2026 | seeds42/123/2026 |

Backbones are frozen and run sequentially; their features are shared by their heads. Each training image contributes one deterministic resize/center-crop view and four random resized-crop/horizontal-flip views. This gives3500 correlated training rows from700 original images. Validation/test use only deterministic center crops. Feature mean/population SD are fitted only on training rows and remain fixed. No TTA, backbone fine-tuning, extra data or pseudo-labeling is used.

MLP: `Linear(dim,128) → BatchNorm → ReLU → Dropout(.3) → Linear(128,20)`. Linear: `Linear(dim,20)`. All heads use AdamW, lr.001, weightdecay.001 and batch64. Each head starts from a fresh seeded initialization and optimizer.

SMLPs retain the historical F0 rule: maximum35epochs, patience7, first strictly best validation accuracy. B/LMLPs train exactly3epochs and retain the final epoch, matching I5. B/Llinear heads train exactly20epochs and retain the final epoch, matching I6. This historical S selection is stated explicitly; the newer fixed B/L schedules do not use validation selection. The final ensemble is exported regardless of its validation improvement.

```
S = mean of S MLP probabilities
B = .75 * mean of B MLP probabilities + .25 * mean of B linear probabilities
L = .75 * mean of L MLP probabilities + .25 * mean of L linear probabilities
final probabilities = (S + B + L) / 3
prediction = argmax(final probabilities)
```

## Outputs and limits

The new run folder contains data hashes, software/hardware/numerical configuration, an execution JSONL log, three freshly extracted feature files, fifteen classifier subfolders (checkpoint/history/probabilities), all-head results, a final training summary, fifteen training-loss plots and the ordered Kaggle CSV. Positive optimizer-step counts and current run IDs are checked for all fifteen classifiers before final export. Test labels are never used. Original split content is checked for duplicates across splits, and the sample submission's string IDs preserve ordering/leading zeros.

The historical0.98376 score is labeled as a reference only; the rerun's score is null until measured on Kaggle. Fresh training may produce different predictions due to software/hardware, numerical flags, augmentation batching or small training differences; it does not automatically inherit the historical score. Pretrained downloads and real T4 execution require Colab/network access.

The notebook demonstrates the complete winning **SSL ensemble**. The assignment also requires an ImageNet-pretrained CNN transfer-learning experiment; the DINOv2 transformer ensemble does not replace that separate requirement.

Targeted synthetic tests execute the notebook's actual S/B/L feature/training stages, all fifteen freshly initialized heads and full ensemble/plot cells with tiny fake frozen encoders. They verify optimizer steps, per-head checkpoint/history agreement, original 3/20epoch B/L schedules, view-major labels, training-only normalization, source-image duplicate rejection, probability weights, sample IDs, provenance and unexecuted/self-contained notebook structure. These are code checks, not actual DINOv2/Kaggle accuracy measurements.
