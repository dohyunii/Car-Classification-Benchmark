"""Fixed two-view probability averaging. No training, tuning or Kaggle upload."""
from pathlib import Path
import argparse
import json
import math
import uuid
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
import accuracy_experiments as j0
import preprocessing_benchmark as common


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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',default='outputs/Flip_TTA')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    common.set_seed(2026)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    data=common.load_data(args.data_root)
    assert (Path(args.data_root)/'sample_submission.csv').exists()
    assert source['record']['model']==j0.MODEL_NAME and source['record']['resolution']==224
    assert source['class_to_idx']==data['class_to_idx']
    assert source['feature_metadata']['manifest']==data['manifest']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=timm.create_model(j0.MODEL_NAME,pretrained=True,num_classes=0,dynamic_img_size=True).to(device)
    backbone.requires_grad_(False).eval()
    assert common.model_fingerprint(backbone)==source['feature_metadata']['weight_sha256']
    head=j0.build_head().to(device);head.load_state_dict(source['head_state_dict'],strict=True)
    mean,std=source['feature_mean'],source['feature_std']
    transform,_=j0.make_transforms(224)
    def loader(paths):return DataLoader(common.ImagePaths(paths,transform),batch_size=args.batch_size,shuffle=False,num_workers=0)
    output=Path(args.output_root)/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output.resolve(),flush=True)
    original,tta=predict_flip_tta(backbone,head,mean,std,loader(data['paths'][1]),device)
    measured=comparison_metrics(data['labels'][1].numpy(),original,tta)
    # Refuse to label a mismatched model as J0. Do not bypass this reproduction gate.
    assert measured['original']['correct']==source['record']['correct']
    assert measured['original']['accuracy']==source['record']['accuracy']
    metadata={'model':'J0 DINOv2-S/14 timm (not F0)', 'checkpoint':str(Path(args.checkpoint).resolve()),
              'checkpoint_sha256':common.file_hash(args.checkpoint),'views':['center_crop','center_crop_horizontal_flip'],
              'combination':'equal probability mean; fixed before evaluation','validation':measured,
              'submitted':False,'test_labels_used':False,'new_training':False,
              'candidate_export_rule':'export original and TTA regardless of validation delta, as requested; neither is claimed better on Kaggle'}
    (output/'summary.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame({'path':[str(p) for p in data['paths'][1]],'true_label':data['labels'][1].numpy(),
                  'original_prediction':original.argmax(1).numpy(),'tta_prediction':tta.argmax(1).numpy()}).to_csv(
                      output/'paired_validation_predictions.csv',index=False)
    np.savez_compressed(output/'validation_probabilities.npz',original=original.numpy(),tta=tta.numpy())
    print('VALIDATION',json.dumps(measured),flush=True)
    original_test,tta_test=predict_flip_tta(backbone,head,mean,std,loader(data['paths'][2]),device)
    save_submission(output/'J0_original_comparison.csv',data['ids'],original_test)
    save_submission(output/'J0_horizontal_flip_TTA_submission.csv',data['ids'],tta_test)
    metadata['test_changed_predictions']=int((original_test.argmax(1)!=tta_test.argmax(1)).sum())
    metadata['prediction_count']=len(data['ids'])
    (output/'summary.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE',output.resolve(),json.dumps(metadata),flush=True)
    return output,metadata

if __name__=='__main__':main()
