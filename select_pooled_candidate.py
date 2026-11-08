"""Audit J3 and export only a predeclared measured-gain candidate. No Kaggle API."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
import accuracy_experiments as j0
import pooled_accuracy_experiment as j3
import preprocessing_benchmark as common


def select_and_export(run,data_root,j0_run):
    run=Path(run)
    summary=json.loads((run/'J3_summary.json').read_text())
    records=sorted(summary['records'],key=lambda r:r['head_seed'])
    assert [r['head_seed'] for r in records]==[42,123,2026]
    baseline=json.loads((Path(j0_run)/'summary.json').read_text())['selection']
    baseline_accuracy=baseline['accuracy']
    assert baseline['head_seed']==2026
    assert np.isfinite(baseline_accuracy) and 0<=baseline_accuracy<=1
    probabilities=[]
    targets=paths=class_names=None
    for record in records:
        assert record['augmentation_seed']==123 and record['resolution']==224 and record['feature_dim']==768
        assert np.isfinite(record['accuracy']) and 0<=record['accuracy']<=1
        name=Path(record['checkpoint']).stem
        z=np.load(run/'metrics'/f'{name}_validation_probabilities.npz',allow_pickle=False)
        if targets is None: targets,paths,class_names=z['target'],z['path'],z['class_names']
        assert targets.shape==paths.shape==(200,) and class_names.shape==(20,)
        assert np.issubdtype(targets.dtype,np.integer) and ((targets>=0)&(targets<20)).all()
        assert len(set(paths.tolist()))==200 and all(isinstance(p,str) and p.strip() for p in paths)
        assert len(set(class_names.tolist()))==20 and all(isinstance(c,str) and c.strip() for c in class_names)
        assert np.array_equal(targets,z['target']) and np.array_equal(paths,z['path'])
        assert np.array_equal(class_names,z['class_names']), 'Probability class columns differ between seeds'
        p=z['probabilities']
        assert p.shape==(200,20) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
        assert ((p>=0)&(p<=1)).all()
        assert float(common.accuracy_score(targets,p.argmax(1)))==record['accuracy']
        probabilities.append(p)
    average=np.mean(np.stack(probabilities),axis=0)
    ensemble_pred=average.argmax(1)
    ensemble={'accuracy':float(common.accuracy_score(targets,ensemble_pred)),
              'macro_f1':float(common.f1_score(targets,ensemble_pred,labels=list(range(20)),average='macro',zero_division=0)),
              'correct':int((targets==ensemble_pred).sum()),'seeds':[42,123,2026],'weights':[1/3]*3}
    single=next(r for r in records if r['head_seed']==2026)
    choice='baseline'
    chosen=[]
    if single['accuracy']>baseline_accuracy: choice='single_2026';chosen=[single]
    if ensemble['accuracy']>max(single['accuracy'],baseline_accuracy): choice='equal_3seed_ensemble';chosen=records
    decision={'baseline':baseline,'pooled_single_2026':single,'pooled_equal_ensemble':ensemble,
              'selection':choice,'validation_count':len(targets),'submitted':False,
              'rule':'single only if accuracy>baseline; ensemble only if strictly>both single and baseline; ties keep fewer heads'}
    # Store every candidate including rejected ones; this is not a subset/weight search.
    (run/'J3_selection.json').write_text(json.dumps(decision,indent=2))
    pd.DataFrame({'path':paths,'true_label':targets,'predicted_label':ensemble_pred}).to_csv(
        run/'metrics'/'J3_equal_ensemble_validation_predictions.csv',index=False)
    if not chosen:
        print('NO_IMPROVEMENT',json.dumps(decision),flush=True)
        return decision
    data=common.load_data(data_root)
    assert np.array_equal(paths,np.asarray([str(p) for p in data['paths'][1]])), 'Validation path order differs from dataset'
    assert np.array_equal(targets,np.asarray(data['labels'][1])), 'Validation target order differs from dataset'
    expected_classes=[name for name,label in sorted(data['class_to_idx'].items(),key=lambda item:item[1])]
    assert np.array_equal(class_names,np.asarray(expected_classes)), 'Probability columns differ from class mapping'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=timm.create_model(j0.MODEL_NAME,pretrained=True,num_classes=0,dynamic_img_size=True).to(device)
    backbone.requires_grad_(False).eval()
    weights_sha=common.model_fingerprint(backbone)
    models=[]
    for record in chosen:
        saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)
        assert saved['record']==record
        assert saved['feature_metadata']['weight_sha256']==weights_sha
        assert saved['feature_metadata']['manifest']==data['manifest']
        assert saved['class_to_idx']==data['class_to_idx']
        head=j3.build_head().to(device)
        head.load_state_dict(saved['head_state_dict'],strict=True);head.eval()
        assert saved['feature_mean'].shape==saved['feature_std'].shape==(768,)
        assert torch.isfinite(saved['feature_mean']).all() and torch.isfinite(saved['feature_std']).all()
        assert (saved['feature_std']>0).all()
        models.append((head,saved['feature_mean'],saved['feature_std']))
    deterministic,_=j0.make_transforms(224)
    loader=DataLoader(common.ImagePaths(data['paths'][2],deterministic),batch_size=8,shuffle=False,num_workers=0)
    predictions=[]
    with torch.inference_mode():
        for images in loader:
            x=j3.pooled_forward(backbone,images.to(device)).float().cpu()
            p=[head(((x-mean)/std).to(device)).softmax(1) for head,mean,std in models]
            predictions.extend(torch.stack(p).mean(0).argmax(1).cpu().tolist())
    assert len(predictions)==len(data['ids']) and all(0<=p<20 for p in predictions)
    frame=pd.DataFrame({'image_id':data['ids'],'label':predictions})
    sample=pd.read_csv(Path(data_root)/'sample_submission.csv',dtype={'image_id':str})
    assert frame.image_id.is_unique and frame.image_id.tolist()==sample.image_id.tolist()
    output=run/'submissions';output.mkdir(exist_ok=True)
    csv=output/f'J3_{choice}_submission.csv'
    frame.to_csv(csv,index=False,mode='x')
    decision['submission']=str(csv)
    decision['checkpoints']=[r['checkpoint'] for r in chosen]
    (run/'J3_selection.json').write_text(json.dumps(decision,indent=2))
    print('SELECTED',json.dumps(decision),flush=True)
    return decision

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--j0-run',required=True)
    parser.add_argument('--data-root',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    select_and_export(args.run,args.data_root,args.j0_run)
