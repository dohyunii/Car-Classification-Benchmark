"""Fixed half F0-three-seed, half I0-base probability blend. No fitting/upload."""
from pathlib import Path
import argparse, gc, hashlib, json, uuid
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

HUB_REF='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
SEEDS=[42,123,2026]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def weight_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build_head(dim):
    return nn.Sequential(nn.Linear(dim,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,20))

class Images(torch.utils.data.Dataset):
    def __init__(self,paths,transform):self.paths,self.transform=paths,transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.paths[index]) as image:return self.transform(image.convert('RGB'))

def load_data(root):
    root=Path(root)
    train,val=[datasets.ImageFolder(root/name) for name in ('train','val')]
    assert len(train)==700 and len(val)==200 and len(train.classes)==20
    assert train.class_to_idx==val.class_to_idx
    assert all(name.startswith(f'{i:03d}_') for i,name in enumerate(train.classes))
    sample=pd.read_csv(root/'sample_submission.csv',dtype={'image_id':str})
    assert {'image_id','label'}<=set(sample.columns) and sample.image_id.is_unique and sample.image_id.notna().all()
    files=sorted(p for p in (root/'test').iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    by_id={p.stem:p for p in files};ids=sample.image_id.tolist()
    assert len(by_id)==len(files) and set(by_id)==set(ids)
    return train.class_to_idx,[Path(p) for p,_ in val.samples],np.asarray(val.targets),[by_id[i] for i in ids],ids

def checked_checkpoint(path,model,dim,seed,mapping):
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['backbone']==model and p['seed_aug']==123 and p['seed_head']==seed
    assert p['class_to_idx']==mapping
    for key in ('feature_mean','feature_std'):
        assert p[key].shape==(dim,) and torch.isfinite(p[key]).all()
    assert (p['feature_std']>0).all()
    expected={'hidden_dim':128,'use_bn':True,'dropout':.3}
    if dim==768:expected={'input_dim':768,**expected,'num_classes':20}
    assert p['head_config']==expected,'Checkpoint is not the requested E4 architecture'
    build_head(dim).load_state_dict(p['head_state_dict'],strict=True)
    return p

@torch.inference_mode()
def infer(backbone,checkpoints,paths,transform,device,batch_size=8):
    backbone.eval();assert not any(p.requires_grad for p in backbone.parameters())
    heads=[]
    for c in checkpoints:
        h=build_head(len(c['feature_mean'])).to(device)
        h.load_state_dict(c['head_state_dict'],strict=True);h.eval()
        heads.append((h,c['feature_mean'].to(device),c['feature_std'].to(device)))
    results=[[] for _ in heads]
    loader=DataLoader(Images(paths,transform),batch_size=batch_size,shuffle=False,num_workers=0)
    for index,images in enumerate(loader):
        features=backbone(images.to(device)).float()
        for j,(head,mean,std) in enumerate(heads):
            assert features.shape==(len(images),len(mean))
            p=head((features-mean)/std).softmax(1)
            assert p.shape==(len(images),20) and torch.isfinite(p).all()
            results[j].append(p.cpu())
        if index%40==0:print('PREDICT_BATCH',index+1,'/',len(loader),flush=True)
    return torch.stack([torch.cat(parts) for parts in results])

def blend_probabilities(small_heads,base):
    assert small_heads.ndim==3 and small_heads.shape[0]==3 and small_heads.shape[2]==20
    assert base.shape==small_heads.shape[1:]
    for value in (small_heads,base):
        assert torch.isfinite(value).all() and ((value>=0)&(value<=1)).all()
        assert torch.allclose(value.sum(-1),torch.ones_like(value.sum(-1)),atol=1e-5)
    small=small_heads.mean(0)
    return small,.5*small+.5*base

def metrics(target,probabilities):
    from sklearn.metrics import accuracy_score,f1_score
    pred=probabilities.argmax(1).numpy()
    return {'accuracy':float(accuracy_score(target,pred)),
        'macro_f1':float(f1_score(target,pred,labels=list(range(20)),average='macro',zero_division=0)),
        'correct':int((pred==target).sum()),'count':len(target)}

def reproduce(measured,expected,name):
    assert measured['correct']==expected['correct'] and abs(measured['accuracy']-expected['accuracy'])<1e-12, (
        f'{name} starting accuracy did not reproduce; diagnose checkpoint/source/preprocessing before exporting')

def export(path,ids,probabilities):
    pred=probabilities.argmax(1).numpy().astype(int)
    assert len(pred)==len(ids)==len(set(ids)) and ((pred>=0)&(pred<20)).all()
    pd.DataFrame({'image_id':list(ids),'label':pred}).to_csv(path,index=False,mode='x')

def run(data_root,f0_run,i0_checkpoint,storage_root,batch_size=8):
    mapping,val_paths,target,test_paths,ids=load_data(data_root)
    f0_run=Path(f0_run);i0_checkpoint=Path(i0_checkpoint)
    summary_path=f0_run/'ensemble_summary.json'
    summary=json.loads(summary_path.read_text())
    assert summary['head_seeds']==SEEDS and summary['augmentation_seed']==123
    assert summary['backbone']=='dinov2_vits14' and summary['hub_ref']==HUB_REF
    assert np.allclose(summary['weights'],[1/3]*3)
    records={r['seed']:r for r in summary['individual_validation']}
    assert set(records)==set(SEEDS)
    paths=[f0_run/Path(records[seed]['checkpoint']).name for seed in SEEDS]
    small=[checked_checkpoint(path,'dinov2_vits14',384,seed,mapping) for path,seed in zip(paths,SEEDS)]
    base=checked_checkpoint(i0_checkpoint,'dinov2_vitb14',768,2026,mapping)
    assert base['hub_ref']==HUB_REF
    source_meta=base['feature_metadata']
    source_backbone=source_meta.get('backbone',source_meta.get('model'))
    assert source_backbone=='dinov2_vitb14','I0 feature metadata identifies a different backbone'
    if 'backbone' in source_meta and 'model' in source_meta:
        assert source_meta['backbone']==source_meta['model'],'Conflicting backbone metadata'
    assert base['feature_metadata']['test_ids']==ids,'Original test ID order differs from I0 source'
    root=Path(data_root).resolve()
    for split in base['feature_metadata']['manifest']:
        # I0 uses a list of entry lists; older helpers used {'split':..., 'files':...}.
        entries=split['files'] if isinstance(split,dict) else split
        assert isinstance(entries,list),'Unrecognized I0 dataset manifest schema'
        for entry in entries:
            image=(root/entry['path']).resolve()
            assert image.is_relative_to(root) and image.is_file()
            assert sha(image)==entry['sha256'],'Dataset image differs from the I0 training/evaluation manifest'
    transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(storage_root)/'student_outputs_F0_I0_Ensemble'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    def load(name,expected_sha,dim):
        model=torch.hub.load(f'facebookresearch/dinov2:{HUB_REF}',name,pretrained=True,trust_repo=True,skip_validation=True).to(device)
        model.requires_grad_(False).eval()
        assert model.embed_dim==dim and weight_digest(model)==expected_sha,'Pretrained backbone differs from checkpoint source'
        return model
    backbone=load('dinov2_vits14',summary['weights_sha256'],384)
    val_small=infer(backbone,small,val_paths,transform,device,batch_size)
    for index,seed in enumerate(SEEDS):reproduce(metrics(target,val_small[index]),records[seed],f'F0 seed{seed}')
    reproduce(metrics(target,val_small.mean(0)),summary['ensemble_validation'],'F0 three-seed ensemble')
    test_small=infer(backbone,small,test_paths,transform,device,batch_size)
    del backbone;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    backbone=load('dinov2_vitb14',base['feature_metadata']['weights_sha256'],768)
    val_base=infer(backbone,[base],val_paths,transform,device,batch_size)[0]
    reproduce(metrics(target,val_base),base['metrics'],'I0 seed2026')
    test_base=infer(backbone,[base],test_paths,transform,device,batch_size)[0]
    val_f0,val_blend=blend_probabilities(val_small,val_base)
    test_f0,test_blend=blend_probabilities(test_small,test_base)
    measured={'F0_three_seed':metrics(target,val_f0),'I0_ViTB14':metrics(target,val_base),'equal_branch_ensemble':metrics(target,val_blend)}
    export(output/'F0_three_seed_control.csv',ids,test_f0)
    export(output/'I0_ViTB14_control.csv',ids,test_base)
    export(output/'F0_I0_equal_branch_ensemble_submission.csv',ids,test_blend)
    pd.DataFrame([{'model':k,**v} for k,v in measured.items()]).to_csv(output/'validation_comparison.csv',index=False)
    np.savez_compressed(output/'validation_probabilities.npz',f0_heads=val_small.numpy(),i0=val_base.numpy(),ensemble=val_blend.numpy(),targets=target)
    provenance={'configuration':'0.5 * mean(F0 seeds42/123/2026 probabilities) + 0.5 * I0 seed2026 probabilities',
        'effective_head_weights':[1/6,1/6,1/6,1/2],'validation':measured,'submitted':False,'trained':False,'TTA':False,
        'test_labels_used':False,'test_prediction_count':len(ids),'checkpoint_sha256':{str(p):sha(p) for p in [*paths,i0_checkpoint]},
        'source_f0_summary_sha256':sha(summary_path),'hub_ref':HUB_REF,
        'selection_rule':'Fixed weights; export regardless of combined validation delta after source reproduction gates pass'}
    (output/'summary.json').write_text(json.dumps(provenance,indent=2))
    print('COMPLETE',output,json.dumps(provenance),flush=True)
    return output,provenance

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','f0-run','i0-checkpoint'):p.add_argument('--'+name,required=True)
    p.add_argument('--storage-root',default='/content');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();run(a.data_root,a.f0_run,a.i0_checkpoint,a.storage_root,a.batch_size)
