"""I4: fixed original/flip probability TTA on the original equal S/B/L ensemble."""
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

from i3_finetune import cached_probabilities, equal_blend


class FlippedTransform:
    def __init__(self,evaluation): self.evaluation=evaluation
    def __call__(self,image): return self.evaluation(image).flip(-1)

def valid_probabilities(value,heads,n):
    assert value.shape==(heads,n,20) and value.dtype==torch.float32
    assert torch.isfinite(value).all() and (value>=0).all() and (value<=1).all()
    assert torch.allclose(value.sum(-1),torch.ones(heads,n),atol=1e-5)

@torch.inference_mode()
def flipped_probabilities(backbone,checkpoints,paths,evaluation,device,batch_size,cache_root,metadata):
    """Resumable branch/split cache; no fitting or validation-based selection."""
    metadata={**metadata,'version':'I4-horizontal-flip-v1','transform':repr(evaluation),
              'torch':str(torch.__version__),'batch_size':batch_size,'device':device}
    signature=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    path=Path(cache_root)/(signature+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        cached=torch.load(path,map_location='cpu',weights_only=True)
        assert cached['metadata']==metadata
        valid_probabilities(cached['probabilities'],len(checkpoints),len(paths))
        print('I4_FLIP_CACHE_HIT',metadata['branch'],metadata['split'],flush=True)
        return cached['probabilities']
    backbone.eval();assert not any(p.requires_grad for p in backbone.parameters())
    heads=[]
    for c in checkpoints:
        head=build_head(len(c['feature_mean'])).to(device)
        head.load_state_dict(c['head_state_dict']);head.eval()
        heads.append((head,c['feature_mean'].to(device),c['feature_std'].to(device)))
    result=[[] for _ in heads]
    loader=DataLoader(Images(paths,FlippedTransform(evaluation)),batch_size=batch_size,shuffle=False,num_workers=0)
    started=time.perf_counter()
    for index,images in enumerate(loader):
        z=backbone(images.to(device)).float()
        for j,(head,mean,std) in enumerate(heads):
            assert z.shape==(len(images),len(mean))
            result[j].append(head((z-mean)/std).softmax(1).cpu())
        if index%10==0 or index+1==len(loader):
            elapsed=time.perf_counter()-started
            print('I4_FLIP',metadata['branch'],metadata['split'],index+1,'/',len(loader),
                'elapsed_min',round(elapsed/60,1),'remaining_min',round(elapsed/(index+1)*(len(loader)-index-1)/60,1),flush=True)
    result=torch.stack([torch.cat(parts) for parts in result])
    valid_probabilities(result,len(checkpoints),len(paths))
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    torch.save({'metadata':metadata,'probabilities':result},temporary);temporary.replace(path)
    return result

def run_i4(data_root,f0_run,i1_run,i2_run,storage_root,small_batch_size=8,large_batch_size=2):
    assert torch.cuda.is_available(), 'Select a GPU runtime in Colab before running I4.'
    assert small_batch_size>=1 and large_batch_size>=1
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
    vs=infer(encoder,small,val_paths,eval224,device,small_batch_size)
    ts=infer(encoder,small,test_paths,eval224,device,small_batch_size)
    for i,seed in enumerate(SEEDS): reproduce(metrics(target,vs[i]),rows[seed],f'F0 seed{seed}')
    reproduce(metrics(target,vs.mean(0)),f0['ensemble_validation'],'F0 ensemble')
    current_val=equal_blend(vs.mean(0),vb,vl);current_test=equal_blend(ts.mean(0),tb,tl)
    reproduce(metrics(target,current_val),ref['validation']['new_equal_S_B_L_blend'],'Current 0.97959 ensemble')
    control_path=i2_run/'F0_B336_L336_equal_backbone_submission.csv'
    control=pd.read_csv(control_path,dtype={'image_id':str})
    assert control.image_id.tolist()==ids and np.array_equal(control.label.to_numpy(),current_test.argmax(1).numpy()), 'Winning submission did not reproduce'
    root=Path(storage_root)/'student_outputs_I4_Ensemble_FlipTTA'
    output=root/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,'DEVICE',torch.cuda.get_device_name(0),flush=True)
    print('Original 0.97959 predictions reproduced. Beginning fixed flip inference.',flush=True)
    flipped={}
    for branch,name,dim,checkpoints,transform,batch_size,digest in [
        ('S224','dinov2_vits14',384,small,eval224,small_batch_size,f0['weights_sha256']),
        ('B336','dinov2_vitb14',768,[base],evaluation,small_batch_size,base['feature_metadata']['weights_sha256']),
        ('L336','dinov2_vitl14',1024,[large],evaluation,large_batch_size,large['feature_metadata']['weights_sha256'])]:
        if branch!='S224':encoder=load_official(name,digest,dim,device)
        pair=[]
        metadata={'branch':branch,'source_checkpoint_sha256':original,
            'hub_ref':HUB_REF,'weights_sha256':digest,'manifest':base['feature_metadata']['manifest'],
            'class_to_idx':mapping,'test_ids':ids}
        for split,images in [('validation',val_paths),('test',test_paths)]:
            pair.append(flipped_probabilities(encoder,checkpoints,images,transform,device,batch_size,root/'flip_cache',{**metadata,'split':split}))
        flipped[branch]=pair
        del encoder;gc.collect();torch.cuda.empty_cache()
    flip_val=equal_blend(flipped['S224'][0].mean(0),flipped['B336'][0][0],flipped['L336'][0][0])
    flip_test=equal_blend(flipped['S224'][1].mean(0),flipped['B336'][1][0],flipped['L336'][1][0])
    candidate_val=.5*(current_val+flip_val);candidate_test=.5*(current_test+flip_test)
    for p in paths:assert sha(p)==original[str(p)],'Source checkpoint modified'
    export(output/'CurrentBest_097959_control.csv',ids,current_test)
    export(output/'I4_S_B336_L336_flip_TTA_submission.csv',ids,candidate_test)
    first=current_val.argmax(1).numpy();second=candidate_val.argmax(1).numpy()
    changed_test=current_test.argmax(1)!=candidate_test.argmax(1)
    report={'experiment':'I4 equal S/B/L original-horizontal-flip probability TTA',
        'rule':'mean original/flip probabilities per head; equal S/B/L branches; F0 heads equal',
        'validation_original':metrics(target,current_val),'validation_TTA':metrics(target,candidate_val),
        'validation_corrected':int(((first!=target)&(second==target)).sum()),
        'validation_regressed':int(((first==target)&(second!=target)).sum()),
        'validation_changed':int((first!=second).sum()),'test_changed':int(changed_test.sum()),
        'source_predictions_reproduced':True,'source_checkpoint_sha256':original,
        'reference_submission_sha256':sha(control_path),'historical_best_kaggle':.97959,
        'export_rule':'always export after integrity checks; validation diagnostic only',
        'small_batch_size':small_batch_size,'large_batch_size':large_batch_size,
        'test_labels_used':False,'training':False,'submitted':False}
    pd.DataFrame({'image_id':ids,'original_label':current_test.argmax(1).numpy(),
        'tta_label':candidate_test.argmax(1).numpy(),'changed':changed_test.numpy()}).to_csv(output/'test_prediction_changes.csv',index=False)
    np.savez_compressed(output/'probabilities.npz',validation_original=current_val.numpy(),validation_tta=candidate_val.numpy(),
        test_original=current_test.numpy(),test_tta=candidate_test.numpy(),targets=target)
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',output,json.dumps(report),flush=True)
    return output,report
