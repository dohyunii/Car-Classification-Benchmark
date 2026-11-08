"""J3: one controlled frozen-DINOv2 CLS+mean-patch representation candidate.

Same J0 224px P3 views, head recipe and seeds. Exact paired CLS-prefix checks
must pass before training; no test features, external data or Kaggle API calls.
"""
from pathlib import Path
import argparse
import json
import platform
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import timm

import accuracy_experiments as j0
import preprocessing_benchmark as common

VERSION = 'J3-v1'
DIMENSION = 768
SEEDS = (42,123,2026)
AUG_SEED = 123
PREFIX_ATOL = 1e-5
TRAIN_CONFIG = dict(j0.TRAIN_CONFIG)


def load_j0_reference(run,data_root):
    run = Path(run)
    if not (run/'summary.json').is_file():
        raise FileNotFoundError('Complete J0 before J3; summary.json is required')
    record = json.loads((run/'metrics'/'J0_S14_224_aug123_head2026.json').read_text())
    assert record['resolution']==224 and record['head_seed']==2026 and record['augmentation_seed']==123
    name = f"J0_S14_224_{record['cache_id']}.pt"
    candidates = [run/'cache'/name,run.parent/'cache'/name]
    paths = [p for p in candidates if p.is_file()]
    if not paths: raise FileNotFoundError(f'Preserve J0 cache {name}; it is required for exact paired-prefix verification')
    reference = torch.load(paths[0],map_location='cpu',weights_only=True)
    meta = reference['metadata']
    assert common.digest(meta)==record['cache_id']
    data = common.load_data(data_root)
    assert (Path(data_root)/'sample_submission.csv').is_file()
    assert meta['model']==j0.MODEL_NAME and meta['size']==224
    assert meta['layout']=='view-major' and meta['original_views']==1 and meta['augmented_views']==4
    assert meta['aug_seed']==123 and meta['manifest']==data['manifest']
    assert meta['class_to_idx']==data['class_to_idx']
    assert reference['X_train'].shape==(len(data['paths'][0])*5,384)
    assert reference['X_val'].shape==(len(data['paths'][1]),384)
    assert torch.equal(reference['y_train'],data['labels'][0].repeat(5))
    assert torch.equal(reference['y_val'],data['labels'][1])
    return reference,data,record,paths[0]


def pooled_forward(model,images):
    """timm ViT forward_features returns final-normalized prefix+patch tokens."""
    tokens = model.forward_features(images)
    assert torch.is_tensor(tokens) and tokens.ndim==3 and tokens.shape[2]==384
    prefix = int(model.num_prefix_tokens)
    assert prefix >= 1 and tokens.shape[1] > prefix
    result = torch.cat([tokens[:,0],tokens[:,prefix:].mean(1)],dim=1)
    assert result.shape==(len(images),768) and torch.isfinite(result).all()
    return result


def verify_prefix(payload,reference):
    for key in ('X_train','X_val'):
        x = payload[key]
        assert x.dtype==torch.float32 and x.shape==(len(reference[key]),768) and torch.isfinite(x).all()
        assert torch.allclose(x[:,:384],reference[key],rtol=0,atol=PREFIX_ATOL), (
            f'J3 {key} CLS prefix differs from paired J0 features; stop and diagnose '
            'source weights, software, extraction batch size, sample order or augmentation RNG. Do not bypass this check.')
    assert torch.equal(payload['y_train'],reference['y_train'])
    assert torch.equal(payload['y_val'],reference['y_val'])


def get_features(model,data,reference,source_cache_id,output,device,batch_size=8,cache_root=None,force=False):
    assert batch_size==reference['metadata']['batch_size'], 'Preserve J0 batch size for paired-prefix control'
    weights = common.model_fingerprint(model)
    assert weights==reference['metadata']['weight_sha256'], 'Backbone weights differ from J0'
    deterministic,augmented = j0.make_transforms(224)
    # This preflight ensures matching source preprocessing before any costly pass.
    assert repr(deterministic)==reference['metadata']['deterministic']
    assert repr(augmented)==reference['metadata']['augmented']
    meta = {'version':VERSION,'model':j0.MODEL_NAME,'pooling':'concat(final_normalized_cls,mean(final_normalized_patch_tokens))',
            'feature_dim':DIMENSION,'pretrained_source':str(model.pretrained_cfg),'weight_sha256':weights,
            'size':224,'aug_seed':AUG_SEED,'original_views':1,'augmented_views':4,'layout':'view-major',
            'deterministic':repr(deterministic),'augmented':repr(augmented),'manifest':data['manifest'],
            'class_to_idx':data['class_to_idx'],'source_j0_cache_id':source_cache_id,
            'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
            'pillow':Image.__version__,'timm':timm.__version__,'device':str(device),'batch_size':batch_size}
    key = common.digest(meta)
    folder = Path(cache_root) if cache_root else Path(output)/'cache'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'J3_S14_224_CLSmeanpatch_{key}.pt'
    if path.is_file() and not force:
        try:
            cached = torch.load(path,map_location='cpu',weights_only=True)
            assert cached['metadata']==meta
            verify_prefix(cached,reference)
            print('J3_CACHE_HIT',path.name,flush=True)
            return cached,key
        except Exception as exc:
            print('J3_CACHE_INVALID',type(exc).__name__,flush=True)
    common.set_seed(AUG_SEED)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    @torch.inference_mode()
    def extract(paths,transform,label):
        loader = DataLoader(common.ImagePaths(paths,transform),batch_size=batch_size,shuffle=False,num_workers=0)
        result=[]
        for index,images in enumerate(loader):
            result.append(pooled_forward(model,images.to(device)).float().cpu())
            if index%40==0: print('J3_EXTRACT',label,min((index+1)*batch_size,len(paths)),'/',len(paths),flush=True)
        return torch.cat(result)
    views = [extract(data['paths'][0],deterministic,'original')]
    for index in range(4): views.append(extract(data['paths'][0],augmented,f'aug{index+1}'))
    payload = {'metadata':meta,'X_train':torch.cat(views),
               'X_val':extract(data['paths'][1],deterministic,'validation'),
               'y_train':data['labels'][0].repeat(5),'y_val':data['labels'][1]}
    verify_prefix(payload,reference)
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)
    print('J3_PREFIX_VERIFIED',PREFIX_ATOL,'cache',path.name,flush=True)
    return payload,key


def build_head():
    return nn.Sequential(nn.Linear(DIMENSION,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,20))


def train_head(features,seed,cache_id,data,output,device):
    name=f'J3_S14_224_CLSmeanpatch_aug123_head{seed}'
    checkpoint_path=Path(output)/'checkpoints'/f'{name}.pt'
    if checkpoint_path.exists(): raise FileExistsError(f'Preserve existing checkpoint: {checkpoint_path}')
    settings = dict(TRAIN_CONFIG)
    mean=features['X_train'].mean(0)
    std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
    train=(features['X_train']-mean)/std
    val=(features['X_val']-mean)/std
    assert settings['batch_size']>1 and len(train)%settings['batch_size']!=1
    common.set_seed(seed)
    head=build_head().to(device)
    loader=DataLoader(TensorDataset(train,features['y_train']),batch_size=settings['batch_size'],shuffle=True,
                      num_workers=0,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(head.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    best,bad,best_epoch,weights=-1.,0,0,None
    history=[]
    for epoch in range(1,settings['epochs']+1):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(x.to(device)),y.to(device))
            loss.backward();optimizer.step();loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode(): pred=head(val.to(device)).argmax(1).cpu()
        accuracy=float(common.accuracy_score(features['y_val'],pred))
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train),'val_accuracy':accuracy})
        if accuracy>best:
            best,bad,best_epoch=accuracy,0,epoch
            weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            bad+=1
            if bad>=settings['patience']: break
    head.load_state_dict(weights);head.eval()
    with torch.inference_mode(): logits=head(val.to(device)).cpu()
    pred=logits.argmax(1)
    record={'experiment':'J3','model':j0.MODEL_NAME,'pooling':features['metadata']['pooling'],
            'resolution':224,'augmentation_seed':AUG_SEED,'head_seed':seed,'feature_dim':DIMENSION,
            'accuracy':best,'correct':int((pred==features['y_val']).sum()),
            'macro_f1':float(common.f1_score(features['y_val'],pred,labels=list(range(20)),average='macro',zero_division=0)),
            'best_epoch':best_epoch,'epochs_run':len(history),'cache_id':cache_id,'training_config':settings}
    record['checkpoint']=str(checkpoint_path)
    with Path(record['checkpoint']).open('xb') as stream:
        torch.save({'head_state_dict':weights,'feature_mean':mean,'feature_std':std,
                    'class_to_idx':data['class_to_idx'],'record':record,'feature_metadata':features['metadata']},stream)
    (Path(output)/'metrics'/f'{name}.json').write_text(json.dumps(record,indent=2))
    pd.DataFrame(history).to_csv(Path(output)/'metrics'/f'{name}_history.csv',index=False)
    pd.DataFrame({'path':[str(p) for p in data['paths'][1]],'true_label':features['y_val'].numpy(),
                  'predicted_label':pred.numpy(),'confidence':logits.softmax(1).max(1).values.numpy()}).to_csv(
                    Path(output)/'metrics'/f'{name}_validation_predictions.csv',index=False)
    # Predeclared downstream ensemble candidate: all three seeds, equal probability
    # average only. Save aligned probabilities; no tuned weights or subset search.
    np.savez_compressed(Path(output)/'metrics'/f'{name}_validation_probabilities.npz',
                        path=np.asarray([str(p) for p in data['paths'][1]]),
                        target=features['y_val'].numpy(),logits=logits.numpy(),
                        probabilities=logits.softmax(1).numpy(),
                        class_names=np.asarray(list(data['class_to_idx'])))
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--j0-run',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',default='outputs/J3')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    common.set_seed(AUG_SEED)
    reference,data,source,source_path=load_j0_reference(args.j0_run,args.data_root)
    output=Path(args.output_root)/uuid.uuid4().hex[:12]
    for directory in ('checkpoints','metrics'):(output/directory).mkdir(parents=True,exist_ok=True)
    print('J3_OUTPUT',output.resolve(),flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=timm.create_model(j0.MODEL_NAME,pretrained=True,num_classes=0,dynamic_img_size=True).to(device)
    model.requires_grad_(False).eval()
    features,key=get_features(model,data,reference,source['cache_id'],output,device,args.batch_size,
                              cache_root=output.parent/'cache',force=args.force)
    records=[]
    for seed in SEEDS:
        record=train_head(features,seed,key,data,output,device)
        records.append(record)
        print('J3_RESULT',json.dumps(record),flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(output/'J3_runs.csv',index=False)
    summary=frame[['accuracy','macro_f1']].agg(['mean','std'])
    summary.to_csv(output/'J3_seed_summary.csv')
    (output/'J3_summary.json').write_text(json.dumps({'records':records,'summary':summary.to_dict(),
        'source_j0_cache':str(source_path),'prefix_verified_atol':PREFIX_ATOL,'no_test_inference':True,
        'no_kaggle_submission':True,'environment':{'python':platform.python_version(),'device':str(device)}},indent=2))
    print('J3_COMPLETE',summary.to_json(),flush=True)


if __name__=='__main__': main()
