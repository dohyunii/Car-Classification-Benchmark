"""I5: three seeds per S/B336/L336 branch; fixed-head training on original caches."""
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

import shutil
from torch.utils.data import TensorDataset
from i3_finetune import cached_probabilities, equal_blend


HEAD_RECIPE={'epochs':3,'batch_size':64,'learning_rate':.001,'weight_decay':.001,
             'optimizer':'AdamW','selection':'final epoch; validation diagnostic only'}

def full_feature_cache(source,storage_root,data_root,target,ids):
    meta=source['feature_metadata'];dim=len(source['feature_mean'])
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    assert signature==source['feature_signature']
    paths=list(Path(storage_root).glob('student_outputs_I*/cache_336/*_'+signature+'.pt'))
    assert len(paths)==1,'Restore the original feature cache: '+signature
    p=torch.load(paths[0],map_location='cpu',weights_only=True)
    assert p['metadata']==meta and meta['test_ids']==ids and meta['image_size']==336
    assert meta['feature_dim']==dim and meta['seed_aug']==123 and meta['n_aug']==4
    assert meta['include_original'] is True and meta['layout']=='view-major'
    training=datasets.ImageFolder(Path(data_root)/'train')
    assert training.class_to_idx==source['class_to_idx'] and len(training)==700
    for key,n in [('X_train',3500),('X_val',len(target)),('X_test',len(ids))]:
        assert p[key].shape==(n,dim) and p[key].dtype==torch.float32 and torch.isfinite(p[key]).all()
    assert torch.equal(p['y_train'],torch.tensor(training.targets).repeat(5))
    assert np.array_equal(p['y_val'].numpy(),target)
    assert torch.allclose(p['X_train'].mean(0),source['feature_mean'],rtol=1e-5,atol=1e-6)
    assert torch.allclose(p['X_train'].std(0,unbiased=False).clamp_min(1e-6),source['feature_std'],rtol=1e-5,atol=1e-6)
    return p,paths[0]

@torch.inference_mode()
def head_probabilities(source,features,device):
    dim=len(source['feature_mean']);head=build_head(dim).to(device)
    head.load_state_dict(source['head_state_dict']);head.eval()
    result=[]
    for key in ('X_val','X_test'):
        x=(features[key]-source['feature_mean'])/source['feature_std']
        p=head(x.to(device)).softmax(1).cpu()
        assert p.shape==(len(x),20) and torch.isfinite(p).all()
        result.append(p)
    return result

def train_seed(seed,source,features,output,device):
    assert seed in (42,123),'Preserve the original 2026 head without retraining'
    assert source['best_epoch']==3,'I5 fixed duration matches the original three-epoch B/L checkpoints'
    assert source['training_config']=={'epochs':35,'batch_size':64,'learning_rate':.001,'weight_decay':.001,'patience':7,'optimizer':'AdamW'}
    mean,std=source['feature_mean'],source['feature_std']
    set_seed(seed);head=build_head(len(mean)).to(device)
    loader=DataLoader(TensorDataset((features['X_train']-mean)/std,features['y_train']),
        batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    history=[]
    for epoch in range(1,HEAD_RECIPE['epochs']+1):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(x.to(device)),y.to(device));assert torch.isfinite(loss)
            loss.backward();optimizer.step();loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode():pv=head(((features['X_val']-mean)/std).to(device)).softmax(1).cpu()
        score=metrics(features['y_val'].numpy(),pv)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(features['X_train']),**score})
        print('I5_HEAD',source['backbone'],'SEED',seed,'EPOCH',epoch,score,flush=True)
    weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    result={**source,'head_state_dict':weights,'seed_head':seed,'selected_epoch':3,
        'metrics':score,'training_config':HEAD_RECIPE,'origin':'I5 new head; final epoch; no validation selection'}
    for key in ('best_epoch','best_accuracy'):result.pop(key,None)
    name=f"I5_{source['backbone']}_336_head{seed}"
    path=Path(output)/(name+'.pt')
    with path.open('xb') as stream:torch.save(result,stream)
    pd.DataFrame(history).to_csv(Path(output)/(name+'_history.csv'),index=False)
    pv,pt=head_probabilities(result,features,device)
    return pv,pt,path,result

def nine_head_blend(small,base,large):
    assert small.ndim==base.ndim==large.ndim==3
    assert small.shape==base.shape==large.shape and small.shape[0]==3 and small.shape[2]==20
    for p in (small,base,large):
        assert torch.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-5)
    return equal_blend(small.mean(0),base.mean(0),large.mean(0))

def run_i5(data_root,f0_run,i1_run,i2_run,storage_root):
    assert torch.cuda.is_available(), 'Select a GPU runtime in Colab before running I5.'
    device='cuda';set_seed(2026)
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
    root=Path(storage_root)/'student_outputs_I5_NineHead'
    output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    branches={};new_paths=[];new_metrics=[];cache_sources={}
    for branch,source,original_v,original_t in [('B336',base,vb,tb),('L336',large,vl,tl)]:
        assert source['best_epoch']==3
        features,cache=full_feature_cache(source,storage_root,data_root,target,ids)
        cache_sources[branch]={'path':str(cache),'signature':source['feature_signature']}
        replay_v,replay_t=head_probabilities(source,features,device)
        assert torch.allclose(replay_v,original_v,atol=2e-5,rtol=2e-4)
        assert torch.allclose(replay_t,original_t,atol=2e-5,rtol=2e-4)
        val_parts=[];test_parts=[]
        for seed in (42,123):
            pv,pt,path,checkpoint=train_seed(seed,source,features,output,device)
            val_parts.append(pv);test_parts.append(pt);new_paths.append(path)
            new_metrics.append({'branch':branch,'seed':seed,**checkpoint['metrics']})
        val_parts.append(original_v);test_parts.append(original_t)
        branches[branch]=(torch.stack(val_parts),torch.stack(test_parts))
        del features;gc.collect()
    candidate_v=nine_head_blend(vs,branches['B336'][0],branches['L336'][0])
    candidate_t=nine_head_blend(ts,branches['B336'][1],branches['L336'][1])
    for path in paths:assert sha(path)==original[str(path)],'Original checkpoint modified'
    for path in paths:
        destination=output/path.name
        with destination.open('xb') as stream, path.open('rb') as source_stream:shutil.copyfileobj(source_stream,stream)
        assert sha(destination)==original[str(path)]
    export(output/'CurrentBest_097959_control.csv',ids,current_test)
    export(output/'I5_S_B336_L336_three_seed_submission.csv',ids,candidate_t)
    changed=current_test.argmax(1)!=candidate_t.argmax(1)
    pd.DataFrame({'image_id':ids,'original_label':current_test.argmax(1).numpy(),
        'candidate_label':candidate_t.argmax(1).numpy(),'changed':changed.numpy()}).to_csv(output/'test_prediction_changes.csv',index=False)
    summary={'experiment':'I5 three seeds per S/B336/L336 backbone','head_seeds':SEEDS,
        'branch_weights':[1/3]*3,'effective_head_weights':[1/9]*9,'head_recipe':HEAD_RECIPE,
        'original_validation':metrics(target,current_val),'candidate_validation':metrics(target,candidate_v),
        'new_heads_validation':new_metrics,'test_changed':int(changed.sum()),'cache_sources':cache_sources,
        'source_checkpoint_sha256':original,'new_checkpoint_sha256':{str(p):sha(p) for p in new_paths},
        'reference_submission_sha256':sha(control_path),'historical_best_kaggle':.97959,
        'source_predictions_reproduced':True,'validation_selection':False,'TTA':False,
        'backbones_frozen':True,'test_labels_used':False,'submitted':False}
    np.savez_compressed(output/'probabilities.npz',validation_original=current_val.numpy(),validation_candidate=candidate_v.numpy(),
        test_original=current_test.numpy(),test_candidate=candidate_t.numpy(),targets=target,
        B336_validation_heads=branches['B336'][0].numpy(),B336_test_heads=branches['B336'][1].numpy(),
        L336_validation_heads=branches['L336'][0].numpy(),L336_test_heads=branches['L336'][1].numpy())
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',output,json.dumps(summary),flush=True)
    return output,summary
