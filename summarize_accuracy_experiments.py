"""Audit and summarize completed J0 runs without retraining or changing selection."""
from pathlib import Path
import argparse
import json
import platform
import pandas as pd
import torch
import torchvision
from PIL import Image
import timm


def summarize(output):
    output=Path(output)
    final=json.loads((output/'summary.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((output/'metrics').glob('J0_S14_*_head*.json'))]
    assert len(records)==6
    frame=pd.DataFrame(records).sort_values(['resolution','head_seed'])
    fingerprints=set()
    for size in (224,336):
        part=frame[frame.resolution==size]
        assert sorted(part.head_seed.tolist())==[42,123,2026]
        assert (part.augmentation_seed==123).all()
    for record in records:
        checkpoint=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)
        fingerprints.add(checkpoint['feature_metadata']['weight_sha256'])
        assert checkpoint['head_state_dict']['0.weight'].shape==(128,384)
        assert checkpoint['feature_mean'].shape==checkpoint['feature_std'].shape==(384,)
        assert checkpoint['record']==record
        name=Path(record['checkpoint']).stem
        hist=pd.read_csv(output/'metrics'/f'{name}_history.csv')
        assert abs(hist.val_accuracy.max()-record['accuracy'])<1e-12
    assert len(fingerprints)==1,'The resolution comparison must share actual pretrained parameters'
    summary=frame.groupby('resolution').agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
    expected=pd.read_csv(output/'metrics'/'J0_summary.csv',index_col='resolution')
    for name in summary.columns:
        assert (abs(expected[name]-summary[name])<1e-12).all()
    env={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':torchvision.__version__,
         'Pillow':Image.__version__,'timm':timm.__version__,'device':final['device'],
         'note':'Initial live process recorded a per-run cache; later CLI maintenance adds shared cache/version fields with unchanged numerical settings.'}
    (output/'environment.json').write_text(json.dumps(env,indent=2))
    paired=[]
    for seed in (42,123,2026):
        first=pd.read_csv(output/'metrics'/f'J0_S14_224_aug123_head{seed}_validation_predictions.csv')
        second=pd.read_csv(output/'metrics'/f'J0_S14_336_aug123_head{seed}_validation_predictions.csv')
        assert first.path.tolist()==second.path.tolist()
        assert first.true_label.tolist()==second.true_label.tolist()
        before=first.predicted_label==first.true_label
        after=second.predicted_label==second.true_label
        paired.append({'head_seed':seed,'corrected_by_336':int((~before&after).sum()),
                       'new_errors_at_336':int((before&~after).sum())})
        changes=pd.DataFrame({'path':first.path,'true_label':first.true_label,
            'prediction_224':first.predicted_label,'prediction_336':second.predicted_label,
            'correct_224':before,'correct_336':after})
        changes.to_csv(output/'metrics'/f'J0_paired_predictions_head{seed}.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'metrics'/'J0_paired_error_changes.csv',index=False)
    delta=100*(summary.loc[336,'accuracy_mean']-summary.loc[224,'accuracy_mean'])
    selected=final['selection']
    checkpoint=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    sample=pd.read_csv(Path(final['data_root'])/'sample_submission.csv',dtype={'image_id':str})
    submission=pd.read_csv(final['submission'],dtype={'image_id':str})
    assert list(submission.columns)==['image_id','label']
    assert submission.image_id.tolist()==sample.image_id.tolist() and submission.image_id.is_unique
    assert pd.api.types.is_integer_dtype(submission.label) and submission.label.between(0,19).all()
    lines=['# J0 measured accuracy comparison','',
        'Actual CPU runs using the provided course train/validation split. No external labeled images, original Stanford Cars labels, pseudo-labels or validation/test training were used. Test inference ran only after resolution selection; no Kaggle submission was made.','',
        '| Resolution | Head seed | Validation accuracy | Macro-F1 | Correct / 200 | Best epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for _,r in frame.iterrows():
        lines.append(f'| {r.resolution} | {r.head_seed} | {r.accuracy:.1%} | {r.macro_f1:.4f} | {r.correct} | {r.best_epoch} |')
    lines+=['','| Resolution | Mean accuracy ± sample SD | Mean macro-F1 ± sample SD |','|---|---:|---:|']
    for size,r in summary.iterrows():
        lines.append(f'| {size} | {100*r.accuracy_mean:.2f}% ± {100*r.accuracy_std:.2f} pp | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} |')
    lines+=['',f'336px changed mean validation accuracy by {delta:+.2f} percentage points against the matched 224px baseline.',
        f"The predeclared mean-accuracy/mean-F1 selection chose {selected['resolution']}px. The submitted-file candidate uses fixed head seed 2026, scoring {selected['accuracy']:.1%} validation accuracy and {selected['macro_f1']:.4f} macro-F1. It was generated locally, **not submitted**.",'',
        'Historical F0 ViT-S/14=98.0% and user-reported I0 ViT-B/14=97.5% were not reverified in these exact implementations. J0 uses the timm port of public self-supervised DINOv2-S weights with dynamic positional interpolation; the controlled 224px baseline is therefore the main comparison. All six checkpoints share the same actual backbone-weight fingerprint.','',
        'Settings: P3 original+four geometric views, augmentation seed123, E4 head, AdamW lr/decay .001, batch64, maximum35 epochs, accuracy-only best checkpoint, patience7. Standardization uses training embeddings only. View count and all head settings remain fixed between resolutions.','',
        'Three seeds reuse one 200-image validation split, so their SD measures training randomness, not uncertainty across new datasets. One validation image changes accuracy by0.5 percentage points. This bounded two-resolution experiment does not establish Kaggle improvement or justify an unrestricted validation search.','',
        f"Submission CSV: `{final['submission']}`. Run directory: `{output}`. Histories, predictions, paired error changes, checkpoints, features, environment metadata and comparison plot are saved there."]
    report=Path(__file__).with_name('ACCURACY_IMPROVEMENT_RESULTS.md')
    report.write_text('\n'.join(lines)+'\n')
    print(frame[['resolution','head_seed','accuracy','macro_f1','correct']].to_string(index=False))
    print(summary.to_string())
    print('Verified checkpoints, histories, common frozen weights, paired predictions and submission order.')
    print('REPORT',report)
    return frame,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();summarize(args.output)
