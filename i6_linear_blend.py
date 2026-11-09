"""I6: fixed linear/MLP mixtures on cached B336/L336 features; preserve I5."""
from pathlib import Path
import gc, hashlib, json, random, shutil, uuid
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from cross_backbone_ensemble import HUB_REF, SEEDS, sha, build_head, load_data, metrics, reproduce, export
from six_head_ensemble import set_seed
from vitb336_experiment import validate_source_data
from i5_nine_head import full_feature_cache, head_probabilities
from i3_finetune import equal_blend

LINEAR_RECIPE={'epochs':20,'batch_size':64,'learning_rate':.001,'weight_decay':.001,
               'optimizer':'AdamW','selection':'final epoch; validation diagnostic only'}
LINEAR_WEIGHT=.25


def validate_probability(p,n):
    assert p.shape==(n,20) and p.dtype==torch.float32 and torch.isfinite(p).all()
    assert (p>=0).all() and (p<=1).all()
    assert torch.allclose(p.sum(1),torch.ones(n),atol=1e-5)


def recover_small(current,base,large):
    for p in (current,base,large):validate_probability(p,len(current))
    small=3*current-base-large
    assert torch.isfinite(small).all() and (small>=-2e-6).all() and (small<=1+2e-6).all()
    assert torch.allclose(small.sum(1),torch.ones(len(small)),atol=2e-5)
    # Subtraction can introduce tiny FP32 negatives for near-zero probabilities.
    small=small.clamp(0,1);small=small/small.sum(1,keepdim=True)
    validate_probability(small,len(current))
    assert torch.allclose(equal_blend(small,base,large),current,atol=2e-6,rtol=1e-5)
    return small


def linear_mlp_blend(small,base,large,linear_base,linear_large):
    for p in (small,base,large,linear_base,linear_large):validate_probability(p,len(small))
    return equal_blend(small,(1-LINEAR_WEIGHT)*base+LINEAR_WEIGHT*linear_base,
        (1-LINEAR_WEIGHT)*large+LINEAR_WEIGHT*linear_large)


@torch.inference_mode()
def linear_probabilities(head,source,features,device,keys=('X_val','X_test')):
    head.eval();result=[]
    for key in keys:
        x=(features[key]-source['feature_mean'])/source['feature_std']
        p=head(x.to(device)).softmax(1).cpu();validate_probability(p,len(x));result.append(p)
    return result


def train_linear(seed,source,features,output,device):
    assert seed in SEEDS
    dim=len(source['feature_mean']);set_seed(seed);head=nn.Linear(dim,20).to(device)
    loader=DataLoader(TensorDataset((features['X_train']-source['feature_mean'])/source['feature_std'],features['y_train']),
        batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(head.parameters(),lr=LINEAR_RECIPE['learning_rate'],weight_decay=LINEAR_RECIPE['weight_decay'])
    history=[]
    for epoch in range(1,LINEAR_RECIPE['epochs']+1):
        head.train();total=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(x.to(device)),y.to(device));assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=loss.item()*len(y)
        pv=linear_probabilities(head,source,features,device,keys=('X_val',))[0]
        score=metrics(features['y_val'].numpy(),pv)
        history.append({'epoch':epoch,'train_loss':total/len(features['X_train']),**score})
        print('I6_LINEAR',source['backbone'],'SEED',seed,'EPOCH',epoch,score,flush=True)
    pv,pt=linear_probabilities(head,source,features,device)
    checkpoint={'head_state_dict':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        'head_config':{'kind':'linear','input_dim':dim,'num_classes':20},'backbone':source['backbone'],
        'hub_ref':HUB_REF,'resolution':336,'seed_head':seed,'seed_aug':123,'selected_epoch':LINEAR_RECIPE['epochs'],
        'class_to_idx':source['class_to_idx'],'feature_mean':source['feature_mean'],'feature_std':source['feature_std'],
        'feature_metadata':source['feature_metadata'],'feature_signature':source['feature_signature'],
        'metrics':score,'training_config':LINEAR_RECIPE}
    name=f"I6_{source['backbone']}_linear_seed{seed}"
    path=Path(output)/(name+'.pt')
    with path.open('xb') as stream:torch.save(checkpoint,stream)
    pd.DataFrame(history).to_csv(Path(output)/(name+'_history.csv'),index=False)
    return pv,pt,path,checkpoint


def load_i5_sources(i5_run,mapping):
    i5_run=Path(i5_run);summary=json.loads((i5_run/'summary.json').read_text())
    assert summary['head_seeds']==SEEDS and np.allclose(summary['effective_head_weights'],[1/9]*9)
    assert summary['TTA'] is False and summary['backbones_frozen'] is True
    expected={Path(p).name:h for p,h in {**summary['source_checkpoint_sha256'],**summary['new_checkpoint_sha256']}.items()}
    assert len(expected)==9
    groups={name:{} for name in ('dinov2_vits14','dinov2_vitb14','dinov2_vitl14')};paths=[]
    for name,digest in expected.items():
        path=i5_run/name;assert sha(path)==digest,'I5 source checkpoint changed: '+name
        c=torch.load(path,map_location='cpu',weights_only=True)
        model=c['backbone'];dim={'dinov2_vits14':384,'dinov2_vitb14':768,'dinov2_vitl14':1024}[model]
        assert c['class_to_idx']==mapping and c['seed_aug']==123 and c['seed_head'] in SEEDS
        assert c['seed_head'] not in groups[model]
        for key in ('feature_mean','feature_std'):assert c[key].shape==(dim,) and torch.isfinite(c[key]).all()
        assert (c['feature_std']>0).all()
        build_head(dim).load_state_dict(c['head_state_dict'],strict=True)
        if dim!=384:assert c['resolution']==336 and c['hub_ref']==HUB_REF
        groups[model][c['seed_head']]=c;paths.append(path)
    for heads in groups.values():assert set(heads)==set(SEEDS)
    return summary,groups,paths,{str(i5_run/name):digest for name,digest in expected.items()}


def run_i6(data_root,i5_run,storage_root):
    set_seed(2026);device='cuda' if torch.cuda.is_available() else 'cpu'
    mapping,_,target,_,ids=load_data(data_root);i5_run=Path(i5_run)
    reference,groups,source_paths,original=load_i5_sources(i5_run,mapping)
    artifacts={str(i5_run/name):sha(i5_run/name) for name in ('summary.json','probabilities.npz','I5_S_B336_L336_three_seed_submission.csv')}
    base=groups['dinov2_vitb14'][2026];large=groups['dinov2_vitl14'][2026]
    validate_source_data(base,data_root,mapping,ids)
    assert large['feature_metadata']['manifest']==base['feature_metadata']['manifest']
    with np.load(i5_run/'probabilities.npz',allow_pickle=False) as saved:
        assert np.array_equal(saved['targets'],target)
        current_v=torch.from_numpy(saved['validation_candidate'].copy())
        current_t=torch.from_numpy(saved['test_candidate'].copy())
        archived={branch:(torch.from_numpy(saved[branch+'_validation_heads'].copy()),torch.from_numpy(saved[branch+'_test_heads'].copy())) for branch in ('B336','L336')}
    validate_probability(current_v,len(target));validate_probability(current_t,len(ids))
    reproduce(metrics(target,current_v),reference['candidate_validation'],'Winning I5 validation')
    control_path=i5_run/'I5_S_B336_L336_three_seed_submission.csv'
    control=pd.read_csv(control_path,dtype={'image_id':str})
    assert control.image_id.tolist()==ids and np.array_equal(control.label.to_numpy(),current_t.argmax(1).numpy()),'I5 CSV/archived probabilities differ'
    root=Path(storage_root)/'student_outputs_I6_LinearBlend';output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,'DEVICE',device,flush=True)
    print('Fixed candidate: S unchanged; B/L each 75% I5 MLP + 25% linear; equal S/B/L branches.',flush=True)
    branches={};linear={};new_paths=[];new_metrics=[];cache_sources={}
    for branch,model,source in [('B336','dinov2_vitb14',base),('L336','dinov2_vitl14',large)]:
        features,cache=full_feature_cache(source,storage_root,data_root,target,ids)
        val_parts=[];test_parts=[]
        for index,seed in enumerate(SEEDS):
            checkpoint=groups[model][seed]
            assert checkpoint['feature_metadata']==source['feature_metadata'] and checkpoint['feature_signature']==source['feature_signature']
            assert torch.equal(checkpoint['feature_mean'],source['feature_mean']) and torch.equal(checkpoint['feature_std'],source['feature_std'])
            pv,pt=head_probabilities(checkpoint,features,device)
            assert archived[branch][0].shape==(3,len(target),20) and archived[branch][1].shape==(3,len(ids),20)
            assert torch.allclose(pv,archived[branch][0][index],atol=2e-5,rtol=2e-4)
            assert torch.allclose(pt,archived[branch][1][index],atol=2e-5,rtol=2e-4)
            reproduce(metrics(target,pv),checkpoint['metrics'],branch+' seed'+str(seed))
            val_parts.append(archived[branch][0][index]);test_parts.append(archived[branch][1][index])
        branches[branch]=(torch.stack(val_parts).mean(0),torch.stack(test_parts).mean(0))
        # All six source heads must replay before any linear training starts.
        cache_sources[branch]={'path':str(cache),'signature':source['feature_signature']}
        del features;gc.collect()
    small_v=recover_small(current_v,branches['B336'][0],branches['L336'][0])
    small_t=recover_small(current_t,branches['B336'][1],branches['L336'][1])
    assert np.array_equal(equal_blend(small_t,branches['B336'][1],branches['L336'][1]).argmax(1).numpy(),control.label.to_numpy())
    print('I5 source replay passed. Beginning fixed linear-head training.',flush=True)
    for branch,source in [('B336',base),('L336',large)]:
        features,_=full_feature_cache(source,storage_root,data_root,target,ids)
        vals=[];tests=[]
        for seed in SEEDS:
            pv,pt,path,checkpoint=train_linear(seed,source,features,output,device)
            vals.append(pv);tests.append(pt);new_paths.append(path)
            new_metrics.append({'branch':branch,'seed':seed,**checkpoint['metrics']})
        linear[branch]=(torch.stack(vals).mean(0),torch.stack(tests).mean(0))
        del features;gc.collect()
    candidate_v=linear_mlp_blend(small_v,branches['B336'][0],branches['L336'][0],linear['B336'][0],linear['L336'][0])
    candidate_t=linear_mlp_blend(small_t,branches['B336'][1],branches['L336'][1],linear['B336'][1],linear['L336'][1])
    for path in source_paths:
        assert sha(path)==original[str(path)],'I5 source checkpoint modified'
        with (output/path.name).open('xb') as stream,path.open('rb') as source_stream:shutil.copyfileobj(source_stream,stream)
        assert sha(output/path.name)==original[str(path)]
    for name,digest in artifacts.items():assert sha(name)==digest,'I5 reference artifact changed'
    export(output/'CurrentBest_098163_control.csv',ids,current_t)
    export(output/'I6_S_B336_L336_MLP_linear_blend_submission.csv',ids,candidate_t)
    changed=current_t.argmax(1)!=candidate_t.argmax(1)
    pd.DataFrame({'image_id':ids,'original_label':current_t.argmax(1).numpy(),'candidate_label':candidate_t.argmax(1).numpy(),'changed':changed.numpy()}).to_csv(output/'test_prediction_changes.csv',index=False)
    summary={'experiment':'I6 fixed MLP/linear head blend','linear_weight_within_B_L':LINEAR_WEIGHT,
        'branch_weights':[1/3]*3,'head_seeds':SEEDS,'linear_recipe':LINEAR_RECIPE,
        'per_head_weights':{'S_MLP':1/9,'B_L_MLP':1/12,'B_L_linear':1/36},
        'original_validation':metrics(target,current_v),'candidate_validation':metrics(target,candidate_v),
        'linear_heads_validation':new_metrics,'test_changed':int(changed.sum()),'cache_sources':cache_sources,
        'source_checkpoint_sha256':original,'source_artifact_sha256':artifacts,'new_checkpoint_sha256':{str(p):sha(p) for p in new_paths},
        'reference_submission_sha256':sha(control_path),'reference_probabilities_sha256':sha(i5_run/'probabilities.npz'),
        'historical_best_kaggle':.98163,'source_predictions_reproduced':True,'validation_selection':False,
        'S_branch':'recovered from saved I5 mean and replayed B/L probabilities; FP32 tolerance checked',
        'TTA':False,'backbones_frozen':True,'test_labels_used':False,'submitted':False}
    np.savez_compressed(output/'probabilities.npz',validation_original=current_v.numpy(),validation_candidate=candidate_v.numpy(),
        test_original=current_t.numpy(),test_candidate=candidate_t.numpy(),targets=target,
        S_validation=small_v.numpy(),S_test=small_t.numpy(),
        B336_MLP_validation=branches['B336'][0].numpy(),B336_MLP_test=branches['B336'][1].numpy(),
        L336_MLP_validation=branches['L336'][0].numpy(),L336_MLP_test=branches['L336'][1].numpy(),
        B336_linear_validation=linear['B336'][0].numpy(),B336_linear_test=linear['B336'][1].numpy(),
        L336_linear_validation=linear['L336'][0].numpy(),L336_linear_test=linear['L336'][1].numpy())
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',output,json.dumps(summary),flush=True)
    return output,summary
