"""J0: paired frozen DINOv2-S resolution experiment. Train split only; no submission API."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import uuid
import platform

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm
import matplotlib.pyplot as plt

import preprocessing_benchmark as common

MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'
SEEDS = (42, 123, 2026)
AUG_SEED = 123
FEATURE_DIM = 384
VERSION = 'J0-v1'
HEAD_CONFIG = {'input':384,'hidden':128,'batchnorm':True,'dropout':0.3,'classes':20}
TRAIN_CONFIG = {'epochs':35,'patience':7,'batch_size':64,'lr':0.001,'weight_decay':0.001}


def make_transforms(size):
    assert size in (224,336)
    normalize = [transforms.ToTensor(), transforms.Normalize(common.MEAN,common.STD)]
    deterministic = transforms.Compose([
        transforms.Resize(size*8//7,interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),*normalize])
    augmented = transforms.Compose([
        transforms.RandomResizedCrop(size,scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),*normalize])
    return deterministic,augmented


def get_features(model,data,size,output,device,batch_size=8,force=False,cache_root=None):
    deterministic,augmented = make_transforms(size)
    metadata = {'version':VERSION,'model':MODEL_NAME,'pretrained_source':str(model.pretrained_cfg),
                'weight_sha256':common.model_fingerprint(model),'size':size,
                'deterministic':repr(deterministic),'augmented':repr(augmented),'aug_seed':AUG_SEED,
                'original_views':1,'augmented_views':4,'layout':'view-major',
                'class_to_idx':data['class_to_idx'],'manifest':data['manifest'],
                # Added versions apply to the next extraction. Already-running J0 processes
                # retain their original keys; this does not change their numeric settings.
                'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                'pillow':Image.__version__,'timm':timm.__version__,'device':str(device),'batch_size':batch_size}
    key = common.digest(metadata)
    cache_directory = Path(cache_root) if cache_root is not None else output/'cache'
    path = cache_directory/f'J0_S14_{size}_{key}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    def valid(payload):
        try:
            assert payload['metadata']==metadata
            for name,count in [('X_train',len(data['paths'][0])*5),('X_val',len(data['paths'][1]))]:
                value=payload[name]
                assert value.dtype==torch.float32 and value.shape==(count,FEATURE_DIM) and torch.isfinite(value).all()
            assert torch.equal(payload['y_train'],data['labels'][0].repeat(5))
            assert torch.equal(payload['y_val'],data['labels'][1])
            return True
        except (KeyError,AssertionError,TypeError,AttributeError): return False
    if path.exists() and not force:
        try:
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if valid(payload):
                print('CACHE_HIT',size,path.name,flush=True)
                return payload,key
            print('CACHE_INVALID',path.name,flush=True)
        except Exception as exc: print('CACHE_UNREADABLE',type(exc).__name__,flush=True)
    common.set_seed(AUG_SEED)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    @torch.inference_mode()
    def extract(paths,transform,label):
        loader=DataLoader(common.ImagePaths(paths,transform),batch_size=batch_size,shuffle=False,num_workers=0)
        values=[]
        for index,batch in enumerate(loader):
            values.append(model(batch.to(device)).float().cpu())
            if index%40==0: print('EXTRACT',size,label,min((index+1)*batch_size,len(paths)),'/',len(paths),flush=True)
        result=torch.cat(values)
        assert result.shape==(len(paths),FEATURE_DIM) and torch.isfinite(result).all()
        return result
    views=[extract(data['paths'][0],deterministic,'original')]
    for index in range(4): views.append(extract(data['paths'][0],augmented,f'aug{index+1}'))
    payload={'metadata':metadata,'X_train':torch.cat(views),
             'X_val':extract(data['paths'][1],deterministic,'validation'),
             'y_train':data['labels'][0].repeat(5),'y_val':data['labels'][1]}
    assert valid(payload)
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)
    return payload,key


def build_head():
    return nn.Sequential(nn.Linear(384,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,20))


def train_head(features,seed,size,cache_id,data,output,device):
    settings=dict(TRAIN_CONFIG)
    assert settings['epochs']>0 and settings['patience']>0 and settings['batch_size']>1
    assert len(features['X_train']) % settings['batch_size'] != 1, 'BatchNorm requires more than one row per training batch'
    mean=features['X_train'].mean(0)
    std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
    train=(features['X_train']-mean)/std
    val=(features['X_val']-mean)/std
    common.set_seed(seed)
    head=build_head().to(device)
    loader=DataLoader(TensorDataset(train,features['y_train']),batch_size=settings['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(head.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    best,bad,best_epoch,weights=-1,0,0,None
    history=[]
    for epoch in range(1,settings['epochs']+1):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(x.to(device)),y.to(device))
            loss.backward();optimizer.step();loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode(): pred=head(val.to(device)).argmax(1).cpu()
        accuracy=common.accuracy_score(features['y_val'],pred)
        history.append({'epoch':epoch,'loss':loss_sum/len(train),'val_accuracy':float(accuracy)})
        if accuracy>best:
            best,bad,best_epoch=float(accuracy),0,epoch
            weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            bad+=1
            if bad>=settings['patience']: break
    head.load_state_dict(weights);head.eval()
    with torch.inference_mode(): logits=head(val.to(device)).cpu()
    pred=logits.argmax(1)
    record={'resolution':size,'head_seed':seed,'augmentation_seed':AUG_SEED,'accuracy':best,
            'macro_f1':float(common.f1_score(features['y_val'],pred,labels=list(range(20)),average='macro',zero_division=0)),
            'correct':int((pred==features['y_val']).sum()),'best_epoch':best_epoch,'epochs_run':len(history),
            'model':MODEL_NAME,'cache_id':cache_id,'head_config':dict(HEAD_CONFIG),'training_config':settings}
    name=f'J0_S14_{size}_aug123_head{seed}'
    record['checkpoint']=str(output/'checkpoints'/f'{name}.pt')
    torch.save({'head_state_dict':weights,'feature_mean':mean,'feature_std':std,'record':record,
                'class_to_idx':data['class_to_idx'],'feature_metadata':features['metadata']},record['checkpoint'])
    (output/'metrics'/f'{name}.json').write_text(json.dumps(record,indent=2))
    pd.DataFrame(history).to_csv(output/'metrics'/f'{name}_history.csv',index=False)
    pd.DataFrame({'path':[str(p) for p in data['paths'][1]],'true_label':features['y_val'].numpy(),
                  'predicted_label':pred.numpy(),'confidence':logits.softmax(1).max(1).values.numpy()}).to_csv(
                       output/'metrics'/f'{name}_validation_predictions.csv',index=False)
    return record


def compare(records,output):
    frame=pd.DataFrame(records).sort_values(['resolution','head_seed'])
    frame.to_csv(output/'metrics'/'J0_runs.csv',index=False)
    summary=frame.groupby('resolution').agg(n=('accuracy','size'),accuracy_mean=('accuracy','mean'),
        accuracy_std=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
    summary.to_csv(output/'metrics'/'J0_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,metric in zip(axes,('accuracy','macro_f1')):
        ax.bar(summary.index.astype(str),summary[metric+'_mean'],yerr=summary[metric+'_std'].fillna(0),capsize=5)
        ax.set(xlabel='Input resolution (px)',ylabel=f'Validation {metric}: mean ± sample SD',ylim=(0.8,1.0))
    fig.suptitle('J0 DINOv2-S + P3/E4; head seeds '+','.join(map(str,SEEDS)))
    fig.tight_layout();fig.savefig(output/'plots'/'J0_seed_comparison.png');plt.close(fig)
    return frame,summary


def generate_submission(model,record,data,output,device,batch_size):
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)
    assert saved['class_to_idx']==data['class_to_idx']
    assert common.model_fingerprint(model)==saved['feature_metadata']['weight_sha256']
    transform,_=make_transforms(record['resolution'])
    head=build_head().to(device)
    head.load_state_dict(saved['head_state_dict']);head.eval();model.eval()
    loader=DataLoader(common.ImagePaths(data['paths'][2],transform),batch_size=batch_size,shuffle=False,num_workers=0)
    predictions=[]
    with torch.inference_mode():
        for batch in loader:
            x=model(batch.to(device)).float().cpu()
            logits=head(((x-saved['feature_mean'])/saved['feature_std']).to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
    assert len(predictions)==len(data['ids']) and all(0<=p<20 for p in predictions)
    frame=pd.DataFrame({'image_id':data['ids'],'label':predictions})
    assert frame.image_id.is_unique and frame.image_id.tolist()==data['ids']
    path=output/'submissions'/f"J0_S14_{record['resolution']}_head{record['head_seed']}_submission.csv"
    frame.to_csv(path,index=False,mode='x')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',default='outputs/J0')
    parser.add_argument('--cache-root',default=None,
                        help='Shared feature cache; defaults to <output-root>/cache, outside UUID run folders.')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    common.set_seed(AUG_SEED)
    output=Path(args.output_root)/uuid.uuid4().hex[:12]
    for name in ('checkpoints','metrics','plots','submissions'): (output/name).mkdir(parents=True,exist_ok=True)
    cache_root=Path(args.cache_root) if args.cache_root else output.parent/'cache'
    print('OUTPUT',str(output.resolve()),flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=common.load_data(args.data_root)
    assert (Path(args.data_root)/'sample_submission.csv').exists()
    # The timm port loads public self-supervised DINOv2 weights, not additional labeled data.
    # Dynamic sizes keep exactly the same pretrained parameters at both resolutions.
    model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0,dynamic_img_size=True).to(device)
    model.requires_grad_(False).eval()
    start=time.perf_counter();records=[]
    for size in (224,336):
        print('START_RESOLUTION',size,flush=True)
        features,key=get_features(model,data,size,output,device,args.batch_size,args.force,cache_root=cache_root)
        for seed in SEEDS:
            record=train_head(features,seed,size,key,data,output,device)
            records.append(record)
            print('RESULT',json.dumps(record),flush=True)
        compare(records,output)
    frame,summary=compare(records,output)
    # Resolution selected by mean accuracy, then mean F1; seed is fixed at 2026, not cherry-picked.
    selected=int(summary.sort_values(['accuracy_mean','macro_f1_mean'],ascending=False,kind='stable').index[0])
    chosen=next(r for r in records if r['resolution']==selected and r['head_seed']==2026)
    print('FINAL_SUMMARY',summary.to_json(),flush=True)
    print('SELECTED',selected,'head seed 2026; no test-driven selection',flush=True)
    path=generate_submission(model,chosen,data,output,device,args.batch_size)
    (output/'summary.json').write_text(json.dumps({'output':str(output.resolve()),'selection':chosen,
        'submission':str(path),'elapsed_seconds':time.perf_counter()-start,'python':platform.python_version(),
        'device':str(device),'no_kaggle_submission':True,'data_root':str(Path(args.data_root).resolve())},indent=2))
    print('COMPLETE',path,flush=True)

if __name__=='__main__': main()
