"""I1: frozen official DINOv2-B/14 at336px; preserve scored224px four-head model."""
from pathlib import Path
import gc,hashlib,json,random,subprocess,uuid
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision
from cross_backbone_ensemble import HUB_REF,SEEDS,sha,weight_digest,load_data,checked_checkpoint,infer,metrics,reproduce,export,build_head,Images
from six_head_ensemble import pinned_backbone,set_seed

SETTINGS={'epochs':35,'batch_size':64,'learning_rate':.001,'weight_decay':.001,'patience':7,'optimizer':'AdamW'}
SIZE=336

def transforms336():
    tail=[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    evaluation=transforms.Compose([transforms.Resize(384,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(336),*tail])
    training=transforms.Compose([transforms.RandomResizedCrop(336,scale=(.8,1.),ratio=(.9,1.1),interpolation=InterpolationMode.BICUBIC),transforms.RandomHorizontalFlip(.5),*tail])
    return evaluation,training

def validate_source_data(base,data_root,mapping,ids):
    meta=base['feature_metadata']
    assert meta.get('backbone',meta.get('model'))=='dinov2_vitb14'
    assert base['hub_ref']==HUB_REF and meta['class_to_idx']==mapping and meta['test_ids']==ids
    assert meta['feature_dim']==768 and meta['seed_aug']==123 and meta['n_aug']==4 and meta['include_original'] is True
    root=Path(data_root).resolve()
    for split in meta['manifest']:
        for entry in (split['files'] if isinstance(split,dict) else split):
            path=(root/entry['path']).resolve()
            assert path.is_relative_to(root) and path.is_file() and sha(path)==entry['sha256'],'Data differs from original I0 run'

def get_features336(backbone,base,data_root,val_paths,target,test_paths,ids,mapping,cache_root,device,batch_size=8,force=False):
    train_ds=datasets.ImageFolder(Path(data_root)/'train')
    assert train_ds.class_to_idx==mapping and len(train_ds)==700
    train_paths=[Path(p) for p,_ in train_ds.samples]
    evaluation,augmentation=transforms336()
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
    fingerprint=weight_digest(backbone)
    assert fingerprint==base['feature_metadata']['weights_sha256']
    meta={'pipeline':'I1-official-B336-v1','backbone':'dinov2_vitb14','hub_ref':HUB_REF,
        'weights_sha256':fingerprint,'image_size':336,'resize_size':384,'feature_dim':768,
        'train_transform':repr(augmentation),'val_transform':repr(evaluation),'seed_aug':123,
        'n_aug':4,'include_original':True,'layout':'view-major','class_to_idx':mapping,
        'manifest':base['feature_metadata']['manifest'],'test_ids':ids,'batch_size':batch_size,
        'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'pillow':Image.__version__,'device':str(device)}
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_root)/f'I1_dinov2_vitb14_336_p3_aug123_{signature}.pt';cache.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        try:
            assert p['metadata']==meta
            for name,n in [('X_train',3500),('X_val',200),('X_test',len(ids))]:
                assert p[name].shape==(n,768) and p[name].dtype==torch.float32 and torch.isfinite(p[name]).all()
            assert torch.equal(p['y_train'],torch.tensor(train_ds.targets).repeat(5))
            assert np.array_equal(p['y_val'].numpy(),target)
            return True
        except (KeyError,AssertionError,TypeError,AttributeError):return False
    if cache.exists() and not force:
        try:
            p=torch.load(cache,map_location='cpu',weights_only=True)
            if valid(p):print('CACHE_HIT',cache,flush=True);return p,signature,cache
            print('INVALID_CACHE; regenerating',cache,flush=True)
        except Exception as e:print('UNREADABLE_CACHE; regenerating',type(e).__name__,flush=True)
    set_seed(123)
    @torch.inference_mode()
    def extract(paths,transform,name):
        loader=DataLoader(Images(paths,transform),batch_size=batch_size,shuffle=False,num_workers=0)
        outputs=[]
        for index,batch in enumerate(loader):
            z=backbone(batch.to(device)).float().cpu()
            assert z.shape==(len(batch),768) and torch.isfinite(z).all()
            outputs.append(z)
            if index%40==0:print('EXTRACT336',name,index+1,'/',len(loader),flush=True)
        return torch.cat(outputs)
    views=[extract(train_paths,evaluation,'original')]
    for index in range(4):views.append(extract(train_paths,augmentation,f'aug{index+1}'))
    p={'metadata':meta,'X_train':torch.cat(views),'X_val':extract(val_paths,evaluation,'validation'),
        'X_test':extract(test_paths,evaluation,'test'),'y_train':torch.tensor(train_ds.targets).repeat(5),'y_val':torch.tensor(target)}
    assert valid(p)
    temporary=cache.with_suffix('.'+uuid.uuid4().hex+'.tmp');torch.save(p,temporary);temporary.replace(cache)
    return p,signature,cache

def train336(features,base,signature,output,device):
    assert base['training_config']==SETTINGS,'Source recipe differs from fixed I1 recipe'
    mean=features['X_train'].mean(0);std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
    train=(features['X_train']-mean)/std;val=(features['X_val']-mean)/std
    set_seed(2026)
    head=build_head(768).to(device)
    loader=DataLoader(TensorDataset(train,features['y_train']),batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(2026))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    best=-1.;bad=0;best_epoch=0;weights=None;history=[]
    for epoch in range(1,36):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(head(x.to(device)),y.to(device));loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode():p=head(val.to(device)).softmax(1).cpu()
        score=metrics(features['y_val'].numpy(),p)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train),**score})
        print('I1_HEAD_EPOCH',epoch,score,flush=True)
        if score['accuracy']>best:
            best,bad,best_epoch=score['accuracy'],0,epoch
            weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            bad+=1
            if bad>=7:break
    head.load_state_dict(weights);head.eval()
    with torch.inference_mode():
        pv=head(val.to(device)).softmax(1).cpu()
        pt=head(((features['X_test']-mean)/std).to(device)).softmax(1).cpu()
    score=metrics(features['y_val'].numpy(),pv)
    path=Path(output)/'I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt'
    checkpoint={'backbone':'dinov2_vitb14','resolution':336,'hub_ref':HUB_REF,'seed_aug':123,'seed_head':2026,
        'head_state_dict':weights,'head_config':base['head_config'],'class_to_idx':base['class_to_idx'],
        'feature_mean':mean,'feature_std':std,'feature_metadata':features['metadata'],'feature_signature':signature,
        'metrics':score,'best_accuracy':best,'best_epoch':best_epoch,'training_config':SETTINGS}
    with path.open('xb') as stream:torch.save(checkpoint,stream)
    pd.DataFrame(history).to_csv(Path(output)/'I1_training_history.csv',index=False)
    return pv,pt,path,checkpoint

def blend336(small_heads,base):
    assert small_heads.ndim==3 and small_heads.shape[0]==3 and small_heads.shape[1:]==base.shape and base.shape[1]==20
    for p in (small_heads,base):
        assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all() and torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-5)
    return .5*small_heads.mean(0)+.5*base

def run336(data_root,f0_run,i0_checkpoint,reference_run,storage_root,batch_size=8,force=False):
    mapping,val_paths,target,test_paths,ids=load_data(data_root)
    f0_run=Path(f0_run);reference_run=Path(reference_run);i0_checkpoint=Path(i0_checkpoint)
    ref=json.loads((reference_run/'summary.json').read_text());summary=json.loads((f0_run/'ensemble_summary.json').read_text())
    assert summary['head_seeds']==SEEDS and summary['hub_ref']==HUB_REF and summary['augmentation_seed']==123
    assert np.allclose(summary['weights'],[1/3]*3)
    rows={r['seed']:r for r in summary['individual_validation']};assert set(rows)==set(SEEDS)
    old_paths=[f0_run/Path(rows[s]['checkpoint']).name for s in SEEDS]+[i0_checkpoint]
    expected={Path(name).name:value for name,value in ref['checkpoint_sha256'].items()};assert len(expected)==4
    originals={str(p):sha(p) for p in old_paths}
    for p in old_paths:assert originals[str(p)]==expected[p.name],'Original source checkpoint differs from scored model'
    small=[checked_checkpoint(p,'dinov2_vits14',384,s,mapping) for p,s in zip(old_paths[:3],SEEDS)]
    base=checked_checkpoint(i0_checkpoint,'dinov2_vitb14',768,2026,mapping)
    validate_source_data(base,data_root,mapping,ids)
    root=Path(storage_root)/'student_outputs_I1_ViTB336';output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    evaluation224=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    backbone=pinned_backbone('dinov2_vits14',summary['weights_sha256'],384,device)
    vs=infer(backbone,small,val_paths,evaluation224,device,batch_size)
    for index,s in enumerate(SEEDS):reproduce(metrics(target,vs[index]),rows[s],f'F0 seed{s}')
    reproduce(metrics(target,vs.mean(0)),summary['ensemble_validation'],'F0 ensemble')
    ts=infer(backbone,small,test_paths,evaluation224,device,batch_size)
    del backbone;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    backbone=pinned_backbone('dinov2_vitb14',base['feature_metadata']['weights_sha256'],768,device)
    vb=infer(backbone,[base],val_paths,evaluation224,device,batch_size)[0]
    reproduce(metrics(target,vb),base['metrics'],'Original I0 224')
    tb=infer(backbone,[base],test_paths,evaluation224,device,batch_size)[0]
    previous=blend336(ts,tb)
    reproduce(metrics(target,blend336(vs,vb)),ref['validation']['equal_branch_ensemble'],'Current four-head model')
    control=pd.read_csv(reference_run/'F0_I0_equal_branch_ensemble_submission.csv',dtype={'image_id':str})
    assert control.image_id.tolist()==ids and np.array_equal(control.label.to_numpy(),previous.argmax(1).numpy()),'Current-best CSV predictions changed'
    features,key,cache=get_features336(backbone,base,data_root,val_paths,target,test_paths,ids,mapping,root/'cache_336',device,batch_size,force)
    pv,pt,path,checkpoint=train336(features,base,key,output,device)
    measured={'current_best_224_blend':metrics(target,blend336(vs,vb)),
        'new_B336_standalone':metrics(target,pv),'new_F0_B336_blend':metrics(target,blend336(vs,pv))}
    for p in old_paths:assert sha(p)==originals[str(p)],'Original checkpoint modified'
    export(output/'CurrentBest_097551_control.csv',ids,previous)
    export(output/'I1_ViTB336_standalone.csv',ids,pt)
    export(output/'F0_I1_336_equal_branch_submission.csv',ids,blend336(ts,pt))
    pd.DataFrame([{'model':name,**value} for name,value in measured.items()]).to_csv(output/'validation_comparison.csv',index=False)
    np.savez_compressed(output/'validation_probabilities.npz',small=vs.numpy(),old_base224=vb.numpy(),new_base336=pv.numpy(),targets=target)
    result={'configuration':'0.5*mean(F0 three heads at224) + 0.5*new I1 B336 seed2026',
        'effective_head_weights':[1/6,1/6,1/6,1/2],'validation':measured,'new_checkpoint':str(path),'new_checkpoint_sha256':sha(path),
        'new_cache':str(cache),'cache_signature':key,'source_checkpoint_sha256':originals,'original_scored_predictions_reproduced':True,
        'augmentation_seed':123,'head_seed':2026,'frozen_backbone':True,'TTA':False,'submitted':False,
        'test_labels_used':False,'prediction_count':len(ids),'historical_reference_Kaggle':.97551,
        'candidate_rule':'one new resolution, fixed50/50blend; export regardless of combined validation delta'}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',output,json.dumps(result),flush=True)
    return output,result
