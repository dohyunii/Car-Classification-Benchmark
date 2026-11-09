"""Package the exact winning I6 ensemble; no training or classifier changes."""
from pathlib import Path
import gc, hashlib, json, random, subprocess, uuid
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from cross_backbone_ensemble import HUB_REF, SEEDS, sha, weight_digest, build_head, Images, export
from six_head_ensemble import set_seed
from vitl336_experiment import load_official

FORMAT='car-ensemble-I6-v1'
SPECS={'dinov2_vits14':(384,224,256),'dinov2_vitb14':(768,336,384),'dinov2_vitl14':(1024,336,384)}


def check_probabilities(p,n):
    assert p.shape==(n,20) and p.dtype==torch.float32 and torch.isfinite(p).all()
    assert (p>=0).all() and (p<=1).all()
    assert torch.allclose(p.sum(1),torch.ones(n),atol=1e-5)


def blend(small,base,large,linear_base,linear_large):
    for p in (small,base,large,linear_base,linear_large):check_probabilities(p,len(small))
    return (small+(.75*base+.25*linear_base)+(.75*large+.25*linear_large))/3


def evaluation_transform(size,resize):
    return transforms.Compose([transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


def make_saved_head(record,dim,device):
    assert record['kind'] in ('mlp','linear')
    head=build_head(dim) if record['kind']=='mlp' else nn.Linear(dim,20)
    head.load_state_dict(record['state_dict'],strict=True)
    return head.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def apply_heads(records,features,dim,device):
    assert features.ndim==2 and features.shape[1]==dim and features.dtype==torch.float32 and torch.isfinite(features).all()
    groups={}
    for kind in ('mlp','linear'):
        selected=sorted((r for r in records if r['kind']==kind),key=lambda r:SEEDS.index(r['seed']))
        if not selected:continue
        assert [r['seed'] for r in selected]==SEEDS
        parts=[]
        for record in selected:
            head=make_saved_head(record,dim,device)
            # Preserve each original head's own normalization exactly.
            normalized=(features-record['mean'])/record['std']
            probabilities=head(normalized.to(device)).softmax(1).cpu()
            check_probabilities(probabilities,len(features));parts.append(probabilities)
        groups[kind]=torch.stack(parts).mean(0)
    return groups


class CarEnsemble:
    """One artifact and one prediction interface for the unchanged I6 model."""
    def __init__(self,path,device=None):
        self.path=Path(path)
        self.bundle=torch.load(self.path,map_location='cpu',weights_only=True)
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._validate()

    def _validate(self):
        b=self.bundle
        assert b['format']==FORMAT and b['hub_ref']==HUB_REF
        assert b['linear_weight']==.25 and b['branch_weights']==[1/3]*3
        assert set(b['models'])==set(SPECS)
        mapping=b['class_to_idx'];assert len(mapping)==20 and sorted(mapping.values())==list(range(20))
        for name,(dim,size,resize) in SPECS.items():
            model=b['models'][name]
            assert (model['dim'],model['size'],model['resize'])==(dim,size,resize)
            expected={('mlp',s) for s in SEEDS}
            if dim!=384:expected|={('linear',s) for s in SEEDS}
            assert len(model['heads'])==len(expected)
            assert {(h['kind'],h['seed']) for h in model['heads']}==expected
            assert len(model['weights_sha256'])==64
            for h in model['heads']:
                assert h['mean'].shape==h['std'].shape==(dim,)
                assert torch.isfinite(h['mean']).all() and torch.isfinite(h['std']).all() and (h['std']>0).all()
                assert all(torch.isfinite(v).all() for v in h['state_dict'].values())
                make_saved_head(h,dim,'cpu')
        reference=b['reference'];ids=reference['ids'];n=len(ids)
        assert n==len(set(ids))==len(reference['labels'])==len(reference['image_sha256'])
        assert all(isinstance(i,str) for i in ids)
        assert reference['labels'].dtype==torch.int64 and ((reference['labels']>=0)&(reference['labels']<20)).all()
        check_probabilities(reference['probabilities'],n)
        assert torch.equal(reference['probabilities'].argmax(1),reference['labels'])
        cache=b['reference_cache'];check_probabilities(cache['S_mean'],n)
        for name,dim in [('B_features',768),('L_features',1024)]:
            assert cache[name].shape==(n,dim) and cache[name].dtype==torch.float32 and torch.isfinite(cache[name]).all()

    @torch.inference_mode()
    def _reference_probabilities(self):
        b=self.bundle;cache=b['reference_cache']
        base=apply_heads(b['models']['dinov2_vitb14']['heads'],cache['B_features'],768,self.device)
        large=apply_heads(b['models']['dinov2_vitl14']['heads'],cache['L_features'],1024,self.device)
        p=blend(cache['S_mean'],base['mlp'],large['mlp'],base['linear'],large['linear'])
        reference=b['reference']
        assert torch.allclose(p,reference['probabilities'],atol=2e-5,rtol=2e-4),'Packaged head probabilities differ from I6'
        assert torch.equal(p.argmax(1),reference['labels']),'Packaged predictions differ from the scored I6 submission'
        return p

    @torch.inference_mode()
    def predict(self,image_paths,ids=None,prefer_saved_features=True,batch_size=8,large_batch_size=2):
        """Return probabilities. Matching original images reuse verified frozen features.

        New images run all three official backbones once each, sequentially.
        Never return stored final labels/probabilities as the prediction result.
        """
        paths=list(map(Path,image_paths));assert paths and batch_size>=1 and large_batch_size>=1
        reference=self.bundle['reference']
        if prefer_saved_features and ids is not None and list(ids)==reference['ids']:
            assert [sha(p) for p in paths]==reference['image_sha256'],'Original test images changed; saved features cannot be reused'
            print('Using verified saved features for the original test images.',flush=True)
            return self._reference_probabilities()
        branches={}
        for name,(dim,size,resize) in SPECS.items():
            spec=self.bundle['models'][name]
            encoder=load_official(name,spec['weights_sha256'],dim,self.device)
            loader=DataLoader(Images(paths,evaluation_transform(size,resize)),batch_size=large_batch_size if dim==1024 else batch_size,shuffle=False,num_workers=0)
            parts=[]
            for index,images in enumerate(loader):
                features=encoder(images.to(self.device)).float().cpu()
                assert features.shape==(len(images),dim) and torch.isfinite(features).all();parts.append(features)
                if index%20==0 or index+1==len(loader):print('PREDICT',name,index+1,'/',len(loader),flush=True)
            features=torch.cat(parts)
            branches[name]=apply_heads(spec['heads'],features,dim,self.device)
            del encoder,features,parts;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        p=blend(branches['dinov2_vits14']['mlp'],branches['dinov2_vitb14']['mlp'],branches['dinov2_vitl14']['mlp'],
            branches['dinov2_vitb14']['linear'],branches['dinov2_vitl14']['linear'])
        if ids is not None and list(ids)==reference['ids']:
            assert [sha(path) for path in paths]==reference['image_sha256']
            assert torch.equal(p.argmax(1),reference['labels']),'Fresh image inference differs from the scored I6 labels; export stopped'
        return p


def test_images(data_root):
    root=Path(data_root)
    sample=pd.read_csv(root/'sample_submission.csv',dtype={'image_id':str})
    assert {'image_id','label'}<=set(sample.columns) and sample.image_id.notna().all() and sample.image_id.is_unique
    ids=sample.image_id.tolist()
    files=[p for p in (root/'test').iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
    by_id={p.stem:p for p in files};assert len(by_id)==len(files) and set(by_id)==set(ids)
    return ids,[by_id[i] for i in ids]


def package_ensemble(i6_run,f0_summary,data_root,bundle_path,device=None):
    """Run once on the user's saved artifacts. Future use needs only the bundle."""
    i6_run=Path(i6_run);bundle_path=Path(bundle_path)
    assert not bundle_path.exists(),'Bundle exists; load it instead of overwriting'
    set_seed(2026);device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    summary=json.loads((i6_run/'summary.json').read_text())
    assert summary['linear_weight_within_B_L']==.25 and summary['branch_weights']==[1/3]*3 and summary['head_seeds']==SEEDS
    assert summary['TTA'] is False and summary['backbones_frozen'] is True
    f0=json.loads(Path(f0_summary).read_text())
    assert f0['backbone']=='dinov2_vits14' and f0['hub_ref']==HUB_REF and f0['head_seeds']==SEEDS
    expected={Path(p).name:h for p,h in {**summary['source_checkpoint_sha256'],**summary['new_checkpoint_sha256']}.items()}
    assert len(expected)==15
    source_hashes={str(i6_run/name):digest for name,digest in expected.items()}
    artifacts={str(p):sha(p) for p in [i6_run/'summary.json',i6_run/'probabilities.npz',i6_run/'I6_S_B336_L336_MLP_linear_blend_submission.csv',Path(f0_summary)]}
    models={name:{'dim':dim,'size':size,'resize':resize,'heads':[]} for name,(dim,size,resize) in SPECS.items()}
    sources={};mapping=None
    for name,digest in expected.items():
        path=i6_run/name;assert sha(path)==digest,'Winning I6 checkpoint changed: '+name
        c=torch.load(path,map_location='cpu',weights_only=True)
        if mapping is None:mapping=c['class_to_idx']
        assert c['class_to_idx']==mapping and c['seed_aug']==123 and c['seed_head'] in SEEDS
        model=c['backbone'];dim=SPECS[model][0]
        kind='linear' if c.get('head_config',{}).get('kind')=='linear' else 'mlp'
        record={'kind':kind,'seed':c['seed_head'],'state_dict':c['head_state_dict'],
            'mean':c['feature_mean'],'std':c['feature_std'],'source_file':name,'source_sha256':digest}
        make_saved_head(record,dim,'cpu');models[model]['heads'].append(record)
        if dim==384:
            assert kind=='mlp';models[model]['weights_sha256']=f0['weights_sha256']
        else:
            assert c['hub_ref']==HUB_REF and c['resolution']==336
            weights=c['feature_metadata']['weights_sha256']
            if 'weights_sha256' in models[model]:assert models[model]['weights_sha256']==weights
            models[model]['weights_sha256']=weights
            if kind=='mlp' and c['seed_head']==2026:sources[model]=c
    ids,paths=test_images(data_root)
    reference_frame=pd.read_csv(i6_run/'I6_S_B336_L336_MLP_linear_blend_submission.csv',dtype={'image_id':str})
    assert reference_frame.image_id.tolist()==ids and len(ids)==700
    mapping_names=sorted(mapping,key=mapping.get);assert len(mapping_names)==20 and all(name.startswith(f'{index:03d}_') for index,name in enumerate(mapping_names))
    base=sources['dinov2_vitb14'];large=sources['dinov2_vitl14']
    assert base['feature_metadata']['test_ids']==large['feature_metadata']['test_ids']==ids
    assert base['feature_metadata']['manifest']==large['feature_metadata']['manifest']
    image_hashes=[sha(p) for p in paths]
    manifest=base['feature_metadata']['manifest'][2]
    entries=manifest['files'] if isinstance(manifest,dict) else manifest
    expected_images={Path(row['path']).stem:row['sha256'] for row in entries}
    assert set(expected_images)==set(ids) and image_hashes==[expected_images[i] for i in ids]
    cached={}
    storage=Path(i6_run).parent.parent
    for model,key in [('dinov2_vitb14','B_features'),('dinov2_vitl14','L_features')]:
        source=sources[model];meta=source['feature_metadata']
        signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        assert signature==source['feature_signature']
        matches=list(storage.glob('student_outputs_I*/cache_336/*_'+signature+'.pt'))
        assert len(matches)==1,'Restore the original source feature cache: '+signature
        payload=torch.load(matches[0],map_location='cpu',weights_only=True)
        assert payload['metadata']==meta
        cached[key]=payload['X_test'].clone()
    with np.load(i6_run/'probabilities.npz',allow_pickle=False) as saved:
        cached['S_mean']=torch.from_numpy(saved['S_test'].copy())
        probabilities=torch.from_numpy(saved['test_candidate'].copy())
    labels=torch.tensor(reference_frame.label.to_numpy(),dtype=torch.int64)
    check_probabilities(probabilities,len(ids));assert torch.equal(probabilities.argmax(1),labels)
    bundle={'format':FORMAT,'hub_ref':HUB_REF,'models':models,'class_to_idx':mapping,
        'linear_weight':.25,'branch_weights':[1/3]*3,
        'reference':{'ids':ids,'image_sha256':image_hashes,'labels':labels,'probabilities':probabilities,
            'reported_kaggle_score':.98376},'reference_cache':cached,
        'provenance':{'source_checkpoint_sha256':source_hashes,'source_artifact_sha256':artifacts,
            'training':False,'TTA':False,'changed_models':False,'changed_weights':False}}
    bundle_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=bundle_path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(bundle,temporary)
        model=CarEnsemble(temporary,device)
        p=model.predict(paths,ids)
        assert torch.equal(p.argmax(1),labels)
        for name,digest in {**source_hashes,**artifacts}.items():assert sha(name)==digest,'Source artifact changed during packaging'
        assert not bundle_path.exists();temporary.replace(bundle_path)
    finally:
        if temporary.exists():temporary.unlink()
    readable={'format':FORMAT,'backbones':[{'name':name,**{k:v for k,v in model.items() if k!='heads'}}|{'heads':[{'kind':h['kind'],'seed':h['seed'],'source_file':h['source_file']} for h in model['heads']]} for name,model in models.items()],
        'formula':'(S_MLP_mean + .75*B_MLP_mean + .25*B_linear_mean + .75*L_MLP_mean + .25*L_linear_mean)/3',
        'hub_ref':HUB_REF,'class_to_idx':mapping,'reference_count':len(ids),'reported_kaggle_score':.98376,
        'original_reference_labels_reproduced':True,'bundle_sha256':sha(bundle_path)}
    (bundle_path.parent/'ensemble_config.json').write_text(json.dumps(readable,indent=2))
    (bundle_path.parent/'bundle_sha256.txt').write_text(sha(bundle_path)+'  '+bundle_path.name+'\n')
    print('BUNDLE_READY',bundle_path,'All',len(ids),'reference labels match.',flush=True)
    return bundle_path


def predict_submission(bundle_path,data_root,output_root,device=None,fresh_image_inference=False):
    """Single prediction entry point; exact-reference gate precedes every CSV export."""
    set_seed(2026);model=CarEnsemble(bundle_path,device)
    ids,paths=test_images(data_root)
    assert ids==model.bundle['reference']['ids'],'Submission order differs from the scored reference'
    probabilities=model.predict(paths,ids,prefer_saved_features=not fresh_image_inference)
    labels=probabilities.argmax(1);reference=model.bundle['reference']['labels']
    assert torch.equal(labels,reference),'Reference labels differ; no CSV exported'
    output=Path(output_root)/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    export(output/'submission.csv',ids,probabilities)
    report={'reference_prediction_count':len(ids),'matching_predictions':int((labels==reference).sum()),
        'changed_predictions':int((labels!=reference).sum()),'all_reference_labels_match':True,
        'prediction_mode':'fresh image inference' if fresh_image_inference else 'verified saved features and unchanged heads',
        'bundle_sha256':sha(bundle_path),'reported_reference_kaggle_score':.98376,
        'new_kaggle_score_measured':False,'training':False,'submitted':False}
    (output/'verification.json').write_text(json.dumps(report,indent=2))
    print('VERIFIED',len(ids),'/',len(ids),'labels; submission:',output/'submission.csv',flush=True)
    return output/'submission.csv',report
