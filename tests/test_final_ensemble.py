import copy,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import final_ensemble as m


class ZeroEncoder(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim
    def forward(self,x):return torch.zeros(len(x),self.dim,device=x.device)


def fixture(root):
    data=root/'data';(data/'test').mkdir(parents=True)
    run=root/'student_outputs_I6_LinearBlend/run';run.mkdir(parents=True)
    ids=[f'{i:06d}' for i in range(700)];rows=[]
    for index,identity in enumerate(ids):
        path=data/'test'/(identity+'.png');Image.new('RGB',(2,2),(index%256,0,0)).save(path)
        rows.append({'path':'test/'+path.name,'sha256':m.sha(path)})
    pd.DataFrame({'image_id':ids,'label':[0]*700}).to_csv(data/'sample_submission.csv',index=False)
    mapping={f'{i:03d}_class':i for i in range(20)};models={};old={};new={};sources={}
    for model,(dim,size,resize) in m.SPECS.items():
        meta={'weights_sha256':'0'*64,'backbone':model,'feature_dim':dim,'test_ids':ids,'manifest':[[],[],rows]}
        signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        models[model]=[]
        for kind in (['mlp'] if dim==384 else ['mlp','linear']):
            for seed in m.SEEDS:
                head=m.build_head(dim) if kind=='mlp' else nn.Linear(dim,20)
                with torch.no_grad():
                    for parameter in head.parameters():parameter.zero_()
                    (head[-1] if kind=='mlp' else head).bias[0]=3.
                c={'head_state_dict':head.state_dict(),'head_config':{'kind':kind},'feature_mean':torch.zeros(dim),
                    'feature_std':torch.ones(dim),'backbone':model,'seed_aug':123,'seed_head':seed,'resolution':size,
                    'class_to_idx':mapping,'hub_ref':m.HUB_REF,'feature_metadata':meta,'feature_signature':signature}
                path=run/f'{model}_{kind}_{seed}.pt';torch.save(c,path)
                (old if kind=='mlp' else new)[str(path)]=m.sha(path)
                models[model].append({'kind':kind,'seed':seed,'state_dict':c['head_state_dict'],'mean':c['feature_mean'],'std':c['feature_std']})
        if dim!=384:
            cache=root/f'student_outputs_I{1 if dim==768 else 2}_cache/cache_336';cache.mkdir(parents=True)
            features=torch.zeros(700,dim)
            torch.save({'metadata':meta,'X_test':features},cache/('features_'+signature+'.pt'))
            sources[model]=m.apply_heads(models[model],features,dim,'cpu')
    s=m.apply_heads(models['dinov2_vits14'],torch.zeros(700,384),384,'cpu')['mlp']
    b=sources['dinov2_vitb14'];l=sources['dinov2_vitl14']
    p=m.blend(s,b['mlp'],l['mlp'],b['linear'],l['linear'])
    m.export(run/'I6_S_B336_L336_MLP_linear_blend_submission.csv',ids,p)
    np.savez_compressed(run/'probabilities.npz',S_test=s.numpy(),test_candidate=p.numpy())
    summary={'linear_weight_within_B_L':.25,'branch_weights':[1/3]*3,'head_seeds':m.SEEDS,'TTA':False,
        'backbones_frozen':True,'source_checkpoint_sha256':old,'new_checkpoint_sha256':new}
    (run/'summary.json').write_text(json.dumps(summary))
    f0=root/'f0_summary.json';f0.write_text(json.dumps({'backbone':'dinov2_vits14','hub_ref':m.HUB_REF,'head_seeds':m.SEEDS,'weights_sha256':'0'*64}))
    return data,run,f0,ids


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def test_package_and_reference_replay_without_experiment_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data,run,f0,ids=fixture(root)
            originals={str(p):m.sha(p) for p in run.iterdir()}
            bundle=root/'final/car_ensemble.pt'
            m.package_ensemble(run,f0,data,bundle,device='cpu')
            for name,digest in originals.items():self.assertEqual(m.sha(name),digest)
            readable=json.loads((bundle.parent/'ensemble_config.json').read_text())
            self.assertEqual(len(readable['backbones']),3);self.assertEqual(sum(len(x['heads']) for x in readable['backbones']),15)
            shutil.rmtree(run)
            for cache in root.glob('student_outputs_I*_cache'):shutil.rmtree(cache)
            f0.unlink()
            with patch.object(m,'load_official',side_effect=AssertionError('saved-feature mode needs no backbone download')):
                csv,report=m.predict_submission(bundle,data,root/'predictions',device='cpu')
            self.assertEqual(report['matching_predictions'],700);self.assertEqual(report['changed_predictions'],0)
            frame=pd.read_csv(csv,dtype={'image_id':str});self.assertEqual(frame.image_id.tolist(),ids)
            self.assertTrue((frame.label==0).all())
            # Generic fresh-image inference remains available from the same bundle.
            model=m.CarEnsemble(bundle,'cpu');paths=[data/'test'/(identity+'.png') for identity in ids[:2]]
            with patch.object(m,'load_official',side_effect=lambda name,sha,dim,device:ZeroEncoder(dim)) as loading:
                p=model.predict(paths)
            self.assertEqual(tuple(p.shape),(2,20));self.assertEqual(loading.call_count,3)
            self.assertTrue((p.argmax(1)==0).all())
            with patch.object(m,'load_official',side_effect=lambda name,sha,dim,device:ZeroEncoder(dim)),patch.object(m,'evaluation_transform',return_value=m.transforms.ToTensor()):
                _,fresh=m.predict_submission(bundle,data,root/'fresh-predictions',device='cpu',fresh_image_inference=True)
            self.assertEqual(fresh['matching_predictions'],700)
            self.assertEqual(fresh['prediction_mode'],'fresh image inference')
            Image.new('RGB',(2,2),'white').save(data/'test'/(ids[0]+'.png'))
            with self.assertRaises(AssertionError):m.predict_submission(bundle,data,root/'rejected',device='cpu')
            self.assertFalse((root/'rejected').exists())
    def test_corrupt_source_refused_and_no_bundle_written(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data,run,f0,_=fixture(root)
            next(run.glob('*.pt')).write_bytes(b'changed')
            bundle=root/'final/car_ensemble.pt'
            with self.assertRaises(AssertionError):m.package_ensemble(run,f0,data,bundle,device='cpu')
            self.assertFalse(bundle.exists())
    def test_reference_mismatch_refused_before_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data,run,f0,_=fixture(root);bundle=root/'final/car_ensemble.pt'
            m.package_ensemble(run,f0,data,bundle,device='cpu')
            b=torch.load(bundle,weights_only=True)
            for model in b['models'].values():
                for h in model['heads']:
                    key='4.bias' if h['kind']=='mlp' else 'bias'
                    h['state_dict'][key].zero_();h['state_dict'][key][1]=20
            torch.save(b,bundle)
            with self.assertRaises(AssertionError):m.predict_submission(bundle,data,root/'rejected',device='cpu')
            self.assertFalse((root/'rejected').exists())
    def test_notebook_self_contained_and_no_training(self):
        path=Path(__file__).resolve().parent.parent/'Final_Car_Classifier.ipynb'
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        scope={};exec(compile(book.cells[3].source,'final-notebook','exec'),scope)
        self.assertTrue(callable(scope['predict_submission']));self.assertTrue(callable(scope['CarEnsemble']))
        self.assertNotIn('from final_ensemble',book.cells[3].source)
        self.assertNotIn('optimizer',book.cells[3].source)
        self.assertNotIn('.backward(',book.cells[3].source)
        for cell in book.cells:
            if cell.cell_type=='code' and '%pip' not in cell.source:compile(cell.source,'final-notebook','exec')

if __name__=='__main__':unittest.main()
