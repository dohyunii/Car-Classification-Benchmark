"""I2: frozen official ViT-L/14 at336; fixed equal S/B/L backbone blend."""
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
from six_head_ensemble import set_seed
from vitb336_experiment import SETTINGS,transforms336,validate_source_data

def load_official(name,expected,dim,device):
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
    assert model.embed_dim==dim
    if expected is not None:assert weight_digest(model)==expected,'Backbone does not match original source weights'
    return model.to(device)

def get_large_features(backbone,base,data_root,val_paths,target,test_paths,ids,mapping,cache_root,device,batch_size=8,force=False):
    train_ds=datasets.ImageFolder(Path(data_root)/'train')
    assert train_ds.class_to_idx==mapping and len(train_ds)==700
    train_paths=[Path(p) for p,_ in train_ds.samples]
    evaluation,augmentation=transforms336()
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
    fingerprint=weight_digest(backbone)
    assert backbone.embed_dim==1024
    meta={'pipeline':'I2-official-L336-v1','backbone':'dinov2_vitl14','hub_ref':HUB_REF,
        'weights_sha256':fingerprint,'image_size':336,'resize_size':384,'feature_dim':1024,
        'train_transform':repr(augmentation),'val_transform':repr(evaluation),'seed_aug':123,
        'n_aug':4,'include_original':True,'layout':'view-major','class_to_idx':mapping,
        'manifest':base['feature_metadata']['manifest'],'test_ids':ids,'batch_size':batch_size,
        'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'pillow':Image.__version__,'device':str(device)}
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_root)/f'I2_dinov2_vitl14_336_p3_aug123_{signature}.pt';cache.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        try:
            assert p['metadata']==meta
            for name,n in [('X_train',3500),('X_val',200),('X_test',len(ids))]:
                assert p[name].shape==(n,1024) and p[name].dtype==torch.float32 and torch.isfinite(p[name]).all()
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
            assert z.shape==(len(batch),1024) and torch.isfinite(z).all()
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

def train_large(features,base,signature,output,device):
    assert base['training_config']==SETTINGS,'Source recipe differs from fixed I2 recipe'
    mean=features['X_train'].mean(0);std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
    train=(features['X_train']-mean)/std;val=(features['X_val']-mean)/std
    set_seed(2026)
    head=build_head(1024).to(device)
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
        print('I2_HEAD_EPOCH',epoch,score,flush=True)
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
    path=Path(output)/'I2_dinov2_vitl14_336_p3_e4_aug123_head2026.pt'
    checkpoint={'backbone':'dinov2_vitl14','resolution':336,'hub_ref':HUB_REF,'seed_aug':123,'seed_head':2026,
        'head_state_dict':weights,'head_config':{'input_dim':1024,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},'class_to_idx':base['class_to_idx'],
        'feature_mean':mean,'feature_std':std,'feature_metadata':features['metadata'],'feature_signature':signature,
        'metrics':score,'best_accuracy':best,'best_epoch':best_epoch,'training_config':SETTINGS}
    with path.open('xb') as stream:torch.save(checkpoint,stream)
    pd.DataFrame(history).to_csv(Path(output)/'I2_training_history.csv',index=False)
    return pv,pt,path,checkpoint

def blend_large(current,large):
    assert current.shape==large.shape and current.ndim==2 and current.shape[1]==20
    for p in (current,large):
        assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all() and torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-5)
    return (2/3)*current+(1/3)*large

def run_large(data_root,f0_run,i1_run,storage_root,large_batch_size=2,control_batch_size=8,force=False):
    assert large_batch_size>=1 and control_batch_size>=1
    mapping,val_paths,target,test_paths,ids=load_data(data_root)
    f0_run=Path(f0_run);i1_run=Path(i1_run)
    source=json.loads((f0_run/'ensemble_summary.json').read_text());reference=json.loads((i1_run/'summary.json').read_text())
    assert source['head_seeds']==SEEDS and source['hub_ref']==HUB_REF and source['augmentation_seed']==123
    assert np.allclose(source['weights'],[1/3]*3)
    rows={r['seed']:r for r in source['individual_validation']};assert set(rows)==set(SEEDS)
    old_paths=[f0_run/Path(rows[s]['checkpoint']).name for s in SEEDS]
    expected={Path(p).name:h for p,h in reference['source_checkpoint_sha256'].items()}
    for p in old_paths:assert sha(p)==expected[p.name],'F0 source differs from scored model'
    base_path=i1_run/Path(reference['new_checkpoint']).name
    assert sha(base_path)==reference['new_checkpoint_sha256'],'B336 source differs from scored model'
    old_paths.append(base_path);originals={str(p):sha(p) for p in old_paths}
    small=[checked_checkpoint(p,'dinov2_vits14',384,s,mapping) for p,s in zip(old_paths[:3],SEEDS)]
    base=checked_checkpoint(base_path,'dinov2_vitb14',768,2026,mapping)
    assert base['resolution']==336 and base['feature_metadata']['image_size']==336
    validate_source_data(base,data_root,mapping,ids)
    root=Path(storage_root)/'student_outputs_I2_ViTL336';output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    evaluation224=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    evaluation336,_=transforms336()
    backbone=load_official('dinov2_vits14',source['weights_sha256'],384,device)
    vs=infer(backbone,small,val_paths,evaluation224,device,control_batch_size)
    for index,s in enumerate(SEEDS):reproduce(metrics(target,vs[index]),rows[s],f'F0 seed{s}')
    reproduce(metrics(target,vs.mean(0)),source['ensemble_validation'],'F0 ensemble')
    ts=infer(backbone,small,test_paths,evaluation224,device,control_batch_size)
    del backbone;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    backbone=load_official('dinov2_vitb14',base['feature_metadata']['weights_sha256'],768,device)
    vb=infer(backbone,[base],val_paths,evaluation336,device,control_batch_size)[0]
    reproduce(metrics(target,vb),base['metrics'],'Preserved B336')
    tb=infer(backbone,[base],test_paths,evaluation336,device,control_batch_size)[0]
    current_val=.5*vs.mean(0)+.5*vb;current_test=.5*ts.mean(0)+.5*tb
    reproduce(metrics(target,current_val),reference['validation']['new_F0_B336_blend'],'Current0.97755blend')
    csv=i1_run/'F0_I1_336_equal_branch_submission.csv'
    old=pd.read_csv(csv,dtype={'image_id':str})
    assert old.image_id.tolist()==ids and np.array_equal(old.label.to_numpy(),current_test.argmax(1).numpy()),'Scored0.97755predictions changed'
    del backbone;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    backbone=load_official('dinov2_vitl14',None,1024,device)
    features,key,cache=get_large_features(backbone,base,data_root,val_paths,target,test_paths,ids,mapping,root/'cache_336',device,large_batch_size,force)
    pv,pt,path,checkpoint=train_large(features,base,key,output,device)
    measured={'current_best_S_B336':metrics(target,current_val),'new_L336_standalone':metrics(target,pv),
        'new_equal_S_B_L_blend':metrics(target,blend_large(current_val,pv))}
    for p in old_paths:assert sha(p)==originals[str(p)],'Original checkpoint modified'
    export(output/'CurrentBest_097755_control.csv',ids,current_test)
    export(output/'I2_ViTL336_standalone.csv',ids,pt)
    export(output/'F0_B336_L336_equal_backbone_submission.csv',ids,blend_large(current_test,pt))
    pd.DataFrame([{'model':name,**value} for name,value in measured.items()]).to_csv(output/'validation_comparison.csv',index=False)
    np.savez_compressed(output/'validation_probabilities.npz',current=current_val.numpy(),large=pv.numpy(),targets=target)
    result={'configuration':'(2/3)*current0.97755ensemble+(1/3)*L336; equal S/B/L backbone branches',
        'effective_head_weights':[1/9,1/9,1/9,1/3,1/3],'validation':measured,'new_checkpoint':str(path),'new_checkpoint_sha256':sha(path),
        'new_cache':str(cache),'cache_signature':key,'source_checkpoint_sha256':originals,'original_scored_predictions_reproduced':True,
        'reference_submission_sha256':sha(csv),'augmentation_seed':123,'head_seed':2026,'frozen_backbone':True,
        'TTA':False,'submitted':False,'test_labels_used':False,'prediction_count':len(ids),'historical_reference_Kaggle':.97755,
        'large_batch_size':large_batch_size,'control_batch_size':control_batch_size,
        'candidate_rule':'one pretrained architecture; fixedweights; export regardless of combined validation delta'}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',output,json.dumps(result),flush=True)
    return output,result
