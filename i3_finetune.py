"""I3: conservative last-two-block B336 fine-tuning; fixed S/B/L blend."""
from pathlib import Path
import gc, hashlib, json, random, subprocess, time, uuid
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from cross_backbone_ensemble import HUB_REF, SEEDS, sha, weight_digest, build_head, Images, load_data, checked_checkpoint, infer, metrics, reproduce, export
from six_head_ensemble import set_seed
from vitb336_experiment import transforms336, validate_source_data
from vitl336_experiment import load_official

RECIPE = {'seed':2026, 'epochs':10, 'patience':3, 'batch_size':4,
          'backbone_lr':2e-6, 'head_lr':3e-5, 'weight_decay':1e-4, 'clip_norm':1.0}

class TrainImages(Dataset):
    def __init__(self, samples, transform): self.samples,self.transform=samples,transform
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        path,label=self.samples[index]
        with Image.open(path) as image: return self.transform(image.convert('RGB')),label

class FineTuneB(nn.Module):
    def __init__(self,backbone,source):
        super().__init__()
        assert backbone.embed_dim==768 and len(backbone.blocks)==12
        self.backbone=backbone
        self.head=build_head(768)
        self.head.load_state_dict(source['head_state_dict'],strict=True)
        self.register_buffer('feature_mean',source['feature_mean'].clone())
        self.register_buffer('feature_std',source['feature_std'].clone())
        assert self.feature_mean.shape==self.feature_std.shape==(768,)
        assert torch.isfinite(self.feature_mean).all() and torch.isfinite(self.feature_std).all() and (self.feature_std>0).all()
        backbone.requires_grad_(False)
        for block in backbone.blocks[-2:]: block.requires_grad_(True)
        backbone.norm.requires_grad_(True)
        self.train(False)
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        self.head.train(mode)
        for module in self.head.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self
    def forward(self,x):
        return self.head((self.backbone(x)-self.feature_mean)/self.feature_std)

@torch.inference_mode()
def image_probabilities(model,paths,transform,device,batch_size=4):
    model.eval(); result=[]
    loader=DataLoader(Images(paths,transform),batch_size=batch_size,shuffle=False,num_workers=0)
    for index,batch in enumerate(loader):
        p=model(batch.to(device)).float().softmax(1).cpu()
        assert p.shape==(len(batch),20) and torch.isfinite(p).all()
        result.append(p)
        if index%40==0: print('I3_INFERENCE',index+1,'/',len(loader),flush=True)
    return torch.cat(result)

@torch.inference_mode()
def cached_probabilities(checkpoint,storage_root,target,ids):
    signature=checkpoint['feature_signature']
    assert signature==hashlib.sha256(json.dumps(checkpoint['feature_metadata'],sort_keys=True).encode()).hexdigest()
    matches=list(Path(storage_root).glob('student_outputs_I*/cache_336/*_'+signature+'.pt'))
    assert len(matches)==1, 'Required original feature cache missing/ambiguous: '+signature+'; restore it from the original run, do not re-extract L336.'
    cached=torch.load(matches[0],map_location='cpu',weights_only=True)
    assert cached['metadata']==checkpoint['feature_metadata']
    assert cached['metadata']['test_ids']==ids and cached['metadata']['image_size']==336
    assert np.array_equal(cached['y_val'].numpy(),target)
    head=build_head(len(checkpoint['feature_mean']));head.load_state_dict(checkpoint['head_state_dict']);head.eval()
    result=[]
    for key,n in [('X_val',len(target)),('X_test',len(ids))]:
        x=cached[key]
        assert x.shape==(n,len(checkpoint['feature_mean'])) and x.dtype==torch.float32 and torch.isfinite(x).all()
        result.append(head((x-checkpoint['feature_mean'])/checkpoint['feature_std']).softmax(1))
    return result

def equal_blend(small,base,large):
    assert small.shape==base.shape==large.shape and small.ndim==2 and small.shape[1]==20
    for p in (small,base,large):
        assert torch.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert torch.allclose(p.sum(1),torch.ones(len(p)),atol=1e-5)
    return (small+base+large)/3

def fine_tune(model,train_loader,val_paths,target,evaluation,device,output,epochs=10):
    assert 1<=epochs<=RECIPE['epochs']
    baseline=image_probabilities(model,val_paths,evaluation,device)
    best_metrics=metrics(target,baseline);best_prob=baseline;best_epoch=0;bad=0
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[{'epoch':0,**best_metrics}]
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':RECIPE['backbone_lr']},
        {'params':model.head.parameters(),'lr':RECIPE['head_lr']}],weight_decay=RECIPE['weight_decay'])
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;started=time.perf_counter()
        for index,(images,labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(images.to(device)),labels.to(device))
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],RECIPE['clip_norm'],error_if_nonfinite=True)
            optimizer.step();loss_sum+=loss.item()*len(labels)
            if index%40==0: print('I3_TRAIN',epoch,index+1,'/',len(train_loader),'loss',loss.item(),flush=True)
        p=image_probabilities(model,val_paths,evaluation,device)
        score=metrics(target,p)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train_loader.dataset),'seconds':time.perf_counter()-started,**score})
        print('I3_EPOCH',epoch,score,flush=True)
        if score['accuracy']>best_metrics['accuracy']:
            best_epoch,best_metrics,best_prob,bad=epoch,score,p,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        pd.DataFrame(history).to_csv(Path(output)/'training_history.csv',index=False)
        if bad>=RECIPE['patience']: break
    model.load_state_dict(best_state,strict=True)
    restored=image_probabilities(model,val_paths,evaluation,device)
    assert torch.allclose(restored,best_prob,atol=1e-6,rtol=1e-5)
    return best_epoch,best_metrics,best_prob,best_state

def run_i3(data_root,f0_run,i1_run,i2_run,storage_root):
    assert torch.cuda.is_available(), 'Select a GPU runtime in Colab before running I3.'
    device='cuda';set_seed(RECIPE['seed'])
    mapping,val_paths,target,test_paths,ids=load_data(data_root)
    f0_run,i1_run,i2_run=map(Path,(f0_run,i1_run,i2_run))
    f0=json.loads((f0_run/'ensemble_summary.json').read_text())
    ref=json.loads((i2_run/'summary.json').read_text())
    assert f0['head_seeds']==SEEDS and f0['hub_ref']==HUB_REF and f0['augmentation_seed']==123
    assert np.allclose(f0['weights'],[1/3]*3)
    assert np.allclose(ref['effective_head_weights'],[1/9,1/9,1/9,1/3,1/3])
    expected={Path(p).name:h for p,h in ref['source_checkpoint_sha256'].items()}
    rows={r['seed']:r for r in f0['individual_validation']}
    small_paths=[f0_run/Path(rows[s]['checkpoint']).name for s in SEEDS]
    base_path=i1_run/'I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt'
    large_path=i2_run/Path(ref['new_checkpoint']).name
    expected[large_path.name]=ref['new_checkpoint_sha256']
    paths=small_paths+[base_path,large_path]
    original={str(p):sha(p) for p in paths}
    for p in paths: assert original[str(p)]==expected[p.name], 'Source checkpoint changed: '+str(p)
    small=[checked_checkpoint(p,'dinov2_vits14',384,s,mapping) for p,s in zip(small_paths,SEEDS)]
    base=checked_checkpoint(base_path,'dinov2_vitb14',768,2026,mapping)
    large=torch.load(large_path,map_location='cpu',weights_only=True)
    assert large['backbone']=='dinov2_vitl14' and large['resolution']==336 and large['class_to_idx']==mapping
    assert base['resolution']==336 and base['feature_metadata']['image_size']==336
    assert large['hub_ref']==HUB_REF and large['feature_metadata']['manifest']==base['feature_metadata']['manifest']
    assert large['feature_mean'].shape==large['feature_std'].shape==(1024,)
    assert torch.isfinite(large['feature_mean']).all() and torch.isfinite(large['feature_std']).all() and (large['feature_std']>0).all()
    validate_source_data(base,data_root,mapping,ids)
    vb,tb=cached_probabilities(base,storage_root,target,ids)
    vl,tl=cached_probabilities(large,storage_root,target,ids)
    reproduce(metrics(target,vb),base['metrics'],'B336 cache')
    reproduce(metrics(target,vl),large['metrics'],'L336 cache')
    evaluation,augmentation=transforms336()
    eval224=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    encoder=load_official('dinov2_vits14',f0['weights_sha256'],384,device)
    vs=infer(encoder,small,val_paths,eval224,device,8)
    ts=infer(encoder,small,test_paths,eval224,device,8)
    for i,seed in enumerate(SEEDS): reproduce(metrics(target,vs[i]),rows[seed],f'F0 seed{seed}')
    reproduce(metrics(target,vs.mean(0)),f0['ensemble_validation'],'F0 ensemble')
    current_val=equal_blend(vs.mean(0),vb,vl);current_test=equal_blend(ts.mean(0),tb,tl)
    reproduce(metrics(target,current_val),ref['validation']['new_equal_S_B_L_blend'],'Current 0.97959 ensemble')
    control_path=i2_run/'F0_B336_L336_equal_backbone_submission.csv'
    control=pd.read_csv(control_path,dtype={'image_id':str})
    assert control.image_id.tolist()==ids and np.array_equal(control.label.to_numpy(),current_test.argmax(1).numpy()), 'Winning submission did not reproduce'
    del encoder;gc.collect();torch.cuda.empty_cache()
    backbone=load_official('dinov2_vitb14',base['feature_metadata']['weights_sha256'],768,device)
    model=FineTuneB(backbone,base).to(device)
    epoch_zero=image_probabilities(model,val_paths,evaluation,device)
    reproduce(metrics(target,epoch_zero),base['metrics'],'Fine-tune epoch zero')
    assert torch.allclose(epoch_zero,vb,atol=2e-5,rtol=2e-4), 'Source image/cache probabilities differ'
    root=Path(storage_root)/'student_outputs_I3_B336_FineTune'
    output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,'DEVICE',torch.cuda.get_device_name(0),flush=True)
    train=datasets.ImageFolder(Path(data_root)/'train');assert train.class_to_idx==mapping and len(train)==700
    set_seed(RECIPE['seed'])
    loader=DataLoader(TrainImages(train.samples,augmentation),batch_size=RECIPE['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(RECIPE['seed']))
    provenance={'recipe':RECIPE,'source_sha256':original,'reference_submission_sha256':sha(control_path),
        'selection':'strict validation accuracy improvement; epoch zero eligible',
        'unfrozen_blocks':[10,11],'final_norm_trainable':True,'head_batchnorm_running_stats':'fixed',
        'feature_statistics':'fixed source training-only statistics','train_count':700,
        'torch':str(torch.__version__),'device':torch.cuda.get_device_name(0),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (output/'configuration.json').write_text(json.dumps(provenance,indent=2))
    epoch,score,pv,state=fine_tune(model,loader,val_paths,target,evaluation,device,output)
    result={'best_epoch':epoch,'baseline_B336':metrics(target,vb),'selected_B336':score,
        'current_ensemble':metrics(target,current_val),'candidate_exported':epoch>0,
        'historical_best_kaggle':.97959,'submitted':False,'test_labels_used':False,
        'source_predictions_reproduced':True,'configuration':provenance}
    export(output/'CurrentBest_097959_control.csv',ids,current_test)
    if epoch>0:
        checkpoint=output/'I3_B336_finetuned.pt'
        with checkpoint.open('xb') as stream:
            torch.save({'model_state_dict':state,'class_to_idx':mapping,'head_config':base['head_config'],
                'backbone':'dinov2_vitb14','resolution':336,'hub_ref':HUB_REF,'best_epoch':epoch,
                'metrics':score,'source_feature_metadata':base['feature_metadata'],'provenance':provenance},stream)
        pt=image_probabilities(model,test_paths,evaluation,device)
        candidate=equal_blend(vs.mean(0),pv,vl)
        result['candidate_ensemble']=metrics(target,candidate)
        result['checkpoint']=str(checkpoint);result['checkpoint_sha256']=sha(checkpoint)
        export(output/'F0_I3_B336_L336_equal_backbone_submission.csv',ids,equal_blend(ts.mean(0),pt,tl))
        np.savez_compressed(output/'probabilities.npz',validation=pv.numpy(),test=pt.numpy(),targets=target)
    else: print('NO_IMPROVEMENT: epoch zero retained; keep the 0.97959 submission.',flush=True)
    for p in paths: assert sha(p)==original[str(p)], 'Source checkpoint modified'
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',output,json.dumps(result),flush=True)
    return output,result
