"""Preserve the scored four-head model; add I0 heads42/123 and fixed six-head blend."""
from pathlib import Path
import argparse, gc, hashlib, json, random, subprocess, time, uuid
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms
from torchvision.transforms import InterpolationMode
from cross_backbone_ensemble import HUB_REF,SEEDS,sha,weight_digest,load_data,checked_checkpoint,infer,metrics,reproduce,export,build_head

SETTINGS={'epochs':35,'batch_size':64,'learning_rate':.001,'weight_decay':.001,'patience':7,'optimizer':'AdamW'}

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def pinned_backbone(name,expected,dim,device):
    try:
        model=torch.hub.load(f'facebookresearch/dinov2:{HUB_REF}',name,pretrained=True,trust_repo=True,skip_validation=True)
    except HTTPError as exc:
        if exc.code not in (502,503,504):raise
        root=Path(torch.hub.get_dir())/('dinov2_git_'+HUB_REF[:12])
        if not root.exists():
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/facebookresearch/dinov2.git',str(root)],check=True)
        subprocess.run(['git','-C',str(root),'checkout','--detach',HUB_REF],check=True)
        assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==HUB_REF
        model=torch.hub.load(str(root),name,source='local',pretrained=True)
    model.requires_grad_(False).eval()
    assert model.embed_dim==dim and weight_digest(model)==expected,'Backbone does not match original source weights'
    return model.to(device)

def feature_cache(base,cache_path,root,mapping,target,ids):
    p=torch.load(cache_path,map_location='cpu',weights_only=True)
    assert p['metadata']==base['feature_metadata']
    meta=p['metadata']
    assert meta.get('backbone',meta.get('model'))=='dinov2_vitb14'
    assert meta['seed_aug']==123 and meta['n_aug']==4 and meta['include_original'] is True
    assert meta['layout']=='view-major' and meta['feature_dim']==768
    signature=hashlib.sha256(json.dumps(p['metadata'],sort_keys=True).encode()).hexdigest()
    assert signature==base['feature_signature']
    train=datasets.ImageFolder(Path(root)/'train')
    assert train.class_to_idx==mapping and len(train)==700
    for name,count in [('X_train',3500),('X_val',200),('X_test',len(ids))]:
        x=p[name];assert x.shape==(count,768) and x.dtype==torch.float32 and torch.isfinite(x).all()
    assert torch.equal(p['y_train'],torch.tensor(train.targets).repeat(5))
    assert np.array_equal(p['y_val'].numpy(),target)
    assert p['metadata']['test_ids']==ids
    assert torch.allclose(p['X_train'].mean(0),base['feature_mean'],rtol=1e-5,atol=1e-6)
    assert torch.allclose(p['X_train'].std(0,unbiased=False).clamp_min(1e-6),base['feature_std'],rtol=1e-5,atol=1e-6)
    for split in p['metadata']['manifest']:
        entries=split['files'] if isinstance(split,dict) else split
        for entry in entries:
            path=(Path(root)/entry['path']).resolve()
            assert path.is_relative_to(Path(root).resolve()) and sha(path)==entry['sha256']
    return p

def train_missing_head(seed,base,features,output,device):
    assert seed in (42,123),'The original2026 head must never be retrained'
    assert base['training_config']==SETTINGS,'Source I0 training recipe differs from the fixed recipe'
    mean,std=base['feature_mean'],base['feature_std']
    train=(features['X_train']-mean)/std;val=(features['X_val']-mean)/std
    set_seed(seed)
    head=build_head(768).to(device)
    loader=DataLoader(TensorDataset(train,features['y_train']),batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    history=[];best=-1.;best_epoch=0;weights=None;bad=0
    for epoch in range(1,36):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(head(x.to(device)),y.to(device));loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode():prob=head(val.to(device)).softmax(1).cpu()
        score=metrics(features['y_val'].numpy(),prob)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train),**score})
        print('I0_HEAD',seed,'EPOCH',epoch,score,flush=True)
        if score['accuracy']>best:
            best,best_epoch,bad=score['accuracy'],epoch,0
            weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            bad+=1
            if bad>=7:break
    head.load_state_dict(weights);head.eval()
    with torch.inference_mode():prob=head(val.to(device)).softmax(1).cpu()
    measured=metrics(features['y_val'].numpy(),prob)
    result={**base,'head_state_dict':weights,'seed_head':seed,'best_epoch':best_epoch,'best_accuracy':best,
            'metrics':{**measured,'seed_head':seed,'seed_aug':123,'best_epoch':best_epoch},
            'origin':'new missing I0 head; original2026 checkpoint unchanged'}
    path=Path(output)/f'I0_dinov2_vitb14_p3_e4_aug123_head{seed}.pt'
    with path.open('xb') as stream:torch.save(result,stream)
    pd.DataFrame(history).to_csv(Path(output)/f'I0_head{seed}_history.csv',index=False)
    return result,path

def average_branches(small,base):
    assert small.ndim==base.ndim==3 and small.shape==base.shape and small.shape[0]==3 and small.shape[2]==20
    for p in (small,base):
        assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all() and torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-5)
    return .5*small.mean(0)+.5*base.mean(0)

def run_six(data_root,f0_run,i0_checkpoint,reference_run,cache_path,storage_root,batch_size=8):
    mapping,val_paths,target,test_paths,ids=load_data(data_root)
    f0_run=Path(f0_run);reference_run=Path(reference_run);i0_checkpoint=Path(i0_checkpoint)
    ref=json.loads((reference_run/'summary.json').read_text())
    source=json.loads((f0_run/'ensemble_summary.json').read_text())
    assert source['head_seeds']==SEEDS and source['augmentation_seed']==123 and source['hub_ref']==HUB_REF
    assert np.allclose(source['weights'],[1/3]*3)
    rows={r['seed']:r for r in source['individual_validation']};assert set(rows)==set(SEEDS)
    old_paths=[f0_run/Path(rows[s]['checkpoint']).name for s in SEEDS]+[i0_checkpoint]
    expected={Path(name).name:value for name,value in ref['checkpoint_sha256'].items()}
    assert len(expected)==4
    original_hashes={str(path):sha(path) for path in old_paths}
    for path in old_paths:assert original_hashes[str(path)]==expected[path.name],'Source checkpoint differs from the scored four-head blend'
    small=[checked_checkpoint(p,'dinov2_vits14',384,s,mapping) for p,s in zip(old_paths[:3],SEEDS)]
    base=checked_checkpoint(i0_checkpoint,'dinov2_vitb14',768,2026,mapping)
    assert base['hub_ref']==HUB_REF
    p=feature_cache(base,cache_path,data_root,mapping,target,ids)
    original_cache_sha=sha(cache_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(storage_root)/'student_outputs_SixHead_Ensemble'/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    extra=[];new_paths=[]
    for seed in (42,123):
        c,path=train_missing_head(seed,base,p,output,device);extra.append(c);new_paths.append(path)
    base_heads=[*extra,base]  #42,123,2026; the source2026 object is reused verbatim.
    transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    model=pinned_backbone('dinov2_vits14',source['weights_sha256'],384,device)
    vs=infer(model,small,val_paths,transform,device,batch_size)
    for i,s in enumerate(SEEDS):reproduce(metrics(target,vs[i]),rows[s],f'F0 seed{s}')
    reproduce(metrics(target,vs.mean(0)),source['ensemble_validation'],'F0 mean')
    ts=infer(model,small,test_paths,transform,device,batch_size)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    model=pinned_backbone('dinov2_vitb14',base['feature_metadata']['weights_sha256'],768,device)
    vb=infer(model,base_heads,val_paths,transform,device,batch_size)
    for i,c in enumerate(base_heads):reproduce(metrics(target,vb[i]),c['metrics'],f'I0 seed{SEEDS[i]}')
    reproduce(metrics(target,.5*vs.mean(0)+.5*vb[2]),ref['validation']['equal_branch_ensemble'],'Scored four-head control')
    tb=infer(model,base_heads,test_paths,transform,device,batch_size)
    old_control=.5*ts.mean(0)+.5*tb[2]
    reference=pd.read_csv(reference_run/'F0_I0_equal_branch_ensemble_submission.csv',dtype={'image_id':str})
    assert reference.image_id.tolist()==ids
    assert np.array_equal(reference.label.to_numpy(),old_control.argmax(1).numpy()),'Reconstructed current-best predictions changed; stop before exporting a new candidate'
    val_new=average_branches(vs,vb);test_new=average_branches(ts,tb)
    export(output/'SixHead_equal_branch_ensemble_submission.csv',ids,test_new)
    export(output/'CurrentBest_four_head_control.csv',ids,old_control)
    export(output/'I0_three_seed_ensemble_control.csv',ids,tb.mean(0))
    measured={'current_best_four_head':metrics(target,.5*vs.mean(0)+.5*vb[2]),
              'I0_three_seed':metrics(target,vb.mean(0)),'six_head_equal_branch':metrics(target,val_new)}
    pd.DataFrame([{'model':name,**value} for name,value in measured.items()]).to_csv(output/'validation_comparison.csv',index=False)
    np.savez_compressed(output/'validation_probabilities.npz',small=vs.numpy(),base=vb.numpy(),six_head=val_new.numpy(),targets=target)
    for path in old_paths:assert sha(path)==original_hashes[str(path)],'Original checkpoint was modified'
    assert sha(cache_path)==original_cache_sha,'Source feature cache was modified'
    summary={'configuration':'0.5*mean(F0 seeds42/123/2026)+0.5*mean(I0 seeds42/123/2026)',
        'effective_head_weights':[1/6]*6,'validation':measured,'source_checkpoint_sha256':original_hashes,
        'new_checkpoint_sha256':{str(path):sha(path) for path in new_paths},'source_cache_sha256':original_cache_sha,
        'original_seed2026_retrained':False,'new_head_seeds':[42,123],'augmentation_seed':123,'submitted':False,
        'TTA':False,'original_scored_csv_reproduced':True,'test_labels_used':False,'prediction_count':len(ids),
        'historical_reference_Kaggle':.97551,'selection':'fixed weights; export irrespective of combined validation delta'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',output,json.dumps(summary),flush=True)
    return output,summary
