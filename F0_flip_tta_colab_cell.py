"""Paste/run inside the existing verified F0 notebook; no repo imports required."""
from pathlib import Path
from types import SimpleNamespace
import uuid, math, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
common=SimpleNamespace(accuracy_score=accuracy_score,f1_score=f1_score)

@torch.inference_mode()
def predict_flip_tta(backbone,head,mean,std,loader,device):
    """Loader must contain deterministic preprocessed images, in required order.

    Returns original probabilities and equal original/horizontal-flip average.
    Works with an already verified F0/I0 backbone and its corresponding head/stats.
    Never passes labels to the model or updates normalization statistics.
    """
    backbone.eval();head.eval()
    assert not any(p.requires_grad for p in backbone.parameters()), 'Use a frozen backbone'
    mean,std=mean.to(device),std.to(device)
    assert mean.ndim==std.ndim==1 and mean.shape==std.shape
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
    original,averaged=[],[]
    for index,batch in enumerate(loader):
        images=batch[0] if isinstance(batch,(list,tuple)) else batch
        images=images.to(device)
        p=[]
        for view in (images,images.flip(-1)):
            features=backbone(view).float()
            assert features.ndim==2 and features.shape[1]==len(mean)
            probabilities=head((features-mean)/std).softmax(1)
            assert probabilities.shape==(len(images),20) and torch.isfinite(probabilities).all()
            p.append(probabilities.cpu())
        original.append(p[0]);averaged.append((p[0]+p[1])*0.5)
        if index%40==0: print('TTA_BATCH',index+1,'/',len(loader),flush=True)
    return torch.cat(original),torch.cat(averaged)

def comparison_metrics(target,original,tta):
    target=np.asarray(target)
    first=original.argmax(1).numpy();second=tta.argmax(1).numpy()
    before=first==target;after=second==target
    corrected=int((~before&after).sum());regressed=int((before&~after).sum())
    n=corrected+regressed
    # Exact paired binomial/McNemar test; small-disagreement samples remain uncertain.
    p_value=min(1.,2*sum(math.comb(n,k) for k in range(min(corrected,regressed)+1))/2**n) if n else 1.
    def score(pred):
        return {'accuracy':float(common.accuracy_score(target,pred)),
                'macro_f1':float(common.f1_score(target,pred,labels=list(range(20)),average='macro',zero_division=0)),
                'correct':int((pred==target).sum()),'count':len(target)}
    return {'original':score(first),'horizontal_flip_tta':score(second),'corrected':corrected,
            'regressed':regressed,'changed_predictions':int((first!=second).sum()),'exact_paired_p_value':p_value}

def save_submission(path,ids,probabilities):
    pred=probabilities.argmax(1).numpy().astype(int)
    assert len(pred)==len(ids)==len(set(ids))
    assert ((pred>=0)&(pred<20)).all()
    pd.DataFrame({'image_id':list(ids),'label':pred}).to_csv(path,index=False,mode='x')

def run_verified_f0_flip_tta(backbone,head,feature_mean,feature_std,val_ds,test_ds,
                             required_ids,output_dir,device,expected_accuracy=0.98,batch_size=16):
    # Use the exact backbone, head and statistics that produced the verified F0 result.
    assert len(val_ds)==200 and len(test_ds)==len(required_ids)
    output=Path(output_dir)/('F0_flip_TTA_'+uuid.uuid4().hex[:10]);output.mkdir(parents=True,exist_ok=False)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    original,tta=predict_flip_tta(backbone,head,feature_mean,feature_std,val_loader,device)
    measured=comparison_metrics(val_ds.targets,original,tta)
    assert abs(measured['original']['accuracy']-expected_accuracy)<1e-12, 'Starting F0 accuracy was not reproduced; diagnose before predicting test'
    print('Validation:',measured)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    original_test,tta_test=predict_flip_tta(backbone,head,feature_mean,feature_std,test_loader,device)
    save_submission(output/'F0_original_comparison.csv',required_ids,original_test)
    save_submission(output/'F0_flip_TTA_submission.csv',required_ids,tta_test)
    (output/'summary.json').write_text(json.dumps({'validation':measured,'submitted':False,
        'combination':'equal mean of original/flip probabilities','source':'existing verified F0 notebook objects',
        'test_prediction_count':len(required_ids),'test_changed_predictions':int((original_test.argmax(1)!=tta_test.argmax(1)).sum())},indent=2))
    print('Candidate files (not submitted):',output)
    return output
