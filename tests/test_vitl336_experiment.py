import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import nbformat
import vitl336_experiment as m

class Encoder(nn.Module):
    def __init__(self,dim):super().__init__();self.embed_dim=dim;self.anchor=nn.Parameter(torch.zeros(1));self.calls=0;self.sizes=[]
    def forward(self,x):
        self.calls+=1;self.sizes.append(x.shape[-1]);return torch.zeros(len(x),self.embed_dim,device=x.device)+self.anchor

class L336Tests(unittest.TestCase):
    def test_weights(self):
        current=torch.zeros(2,20);large=torch.zeros(2,20);current[:,0]=1;large[:,1]=1
        p=m.blend_large(current,large)
        self.assertTrue(torch.allclose(p[:,0],torch.full((2,),2/3)))
        self.assertTrue(torch.allclose(p[:,1],torch.full((2,),1/3)))
        with self.assertRaises(AssertionError):m.blend_large(current,large[:1])
    def test_end_to_end_large_cache_and_original_unchanged(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f0=root/'f0';f0.mkdir();i1=root/'i1';i1.mkdir()
            image=root/'image.png';Image.new('RGB',(32,32),(100,110,120)).save(image)
            mapping={f'{i:03d}_car':i for i in range(20)};ids=['009','001'];target=np.tile(np.arange(20),10);train_target=np.tile(np.arange(20),35)
            models={dim:Encoder(dim).eval().requires_grad_(False) for dim in (384,768,1024)}
            small=[];rows=[];old_paths=[]
            for seed in m.SEEDS:
                torch.manual_seed(seed);head=m.build_head(384).eval()
                c={'backbone':'dinov2_vits14','seed_aug':123,'seed_head':seed,'class_to_idx':mapping,'feature_mean':torch.zeros(384),'feature_std':torch.ones(384),
                   'head_config':{'hidden_dim':128,'use_bn':True,'dropout':.3},'head_state_dict':head.state_dict()}
                path=f0/f'seed{seed}.pt';torch.save(c,path);old_paths.append(path)
                with torch.inference_mode():p=head(torch.zeros(200,384)).softmax(1)
                small.append(p);rows.append({'seed':seed,'checkpoint':str(path),**m.metrics(target,p)})
            fs={'head_seeds':m.SEEDS,'augmentation_seed':123,'hub_ref':m.HUB_REF,'weights':[1/3]*3,'individual_validation':rows,
                'weights_sha256':m.weight_digest(models[384]),'ensemble_validation':m.metrics(target,torch.stack(small).mean(0))}
            (f0/'ensemble_summary.json').write_text(json.dumps(fs))
            head=m.build_head(768).eval()
            with torch.inference_mode():bp=head(torch.zeros(200,768)).softmax(1)
            meta={'backbone':'dinov2_vitb14','feature_dim':768,'image_size':336,'seed_aug':123,'n_aug':4,'include_original':True,'class_to_idx':mapping,'test_ids':ids,
                'weights_sha256':m.weight_digest(models[768]),'manifest':[[],[],[{'path':'image.png','sha256':m.sha(image)}]]}
            base={'backbone':'dinov2_vitb14','resolution':336,'hub_ref':m.HUB_REF,'seed_aug':123,'seed_head':2026,'class_to_idx':mapping,
                'head_config':{'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},'head_state_dict':head.state_dict(),
                'feature_mean':torch.zeros(768),'feature_std':torch.ones(768),'feature_metadata':meta,'training_config':m.SETTINGS,'metrics':m.metrics(target,bp)}
            base_path=i1/'B336.pt';torch.save(base,base_path);old_paths.append(base_path)
            current=.5*torch.stack(small).mean(0)+.5*bp
            ref={'source_checkpoint_sha256':{str(p):m.sha(p) for p in old_paths[:3]},'new_checkpoint':str(base_path),'new_checkpoint_sha256':m.sha(base_path),
                 'validation':{'new_F0_B336_blend':m.metrics(target,current)}}
            (i1/'summary.json').write_text(json.dumps(ref));m.export(i1/'F0_I1_336_equal_branch_submission.csv',ids,current[:2])
            hashes={p:m.sha(p) for p in old_paths}
            fake_train=type('Train',(),{'class_to_idx':mapping,'targets':train_target.tolist(),'samples':[(str(image),int(y)) for y in train_target],'__len__':lambda self:700})()
            data=(mapping,[image]*200,target,[image]*2,ids)
            with patch.object(m,'load_data',return_value=data),patch.object(m.datasets,'ImageFolder',return_value=fake_train),patch.object(m,'load_official',side_effect=lambda name,expected,dim,device:models[dim]):
                output,result=m.run_large(root,f0,i1,root,large_batch_size=2,control_batch_size=8)
                features=torch.load(result['new_cache'],weights_only=True,map_location='cpu')
                point=torch.load(result['new_checkpoint'],weights_only=True,map_location='cpu')
                self.assertEqual(features['X_train'].shape,(3500,1024));self.assertEqual(point['head_state_dict']['0.weight'].shape,(128,1024))
                self.assertEqual(point['backbone'],'dinov2_vitl14');self.assertEqual(point['head_config']['input_dim'],1024)
                self.assertTrue(torch.equal(point['feature_mean'],features['X_train'].mean(0)))
                self.assertTrue(torch.equal(point['feature_std'],features['X_train'].std(0,unbiased=False).clamp_min(1e-6)))
                self.assertTrue(torch.equal(features['y_train'],torch.tensor(train_target).repeat(5)))
                calls=models[1024].calls
                cached,key,path=m.get_large_features(models[1024],base,root,data[1],target,data[3],ids,mapping,output.parent/'cache_336','cpu',2)
                self.assertEqual(calls,models[1024].calls)
                corrupt=dict(cached);corrupt['X_train']=torch.zeros(3500,768);torch.save(corrupt,path)
                with patch('vitl336_experiment.Images',side_effect=RuntimeError('regeneration')):
                    with self.assertRaisesRegex(RuntimeError,'regeneration'):m.get_large_features(models[1024],base,root,data[1],target,data[3],ids,mapping,output.parent/'cache_336','cpu',2)
            for p,h in hashes.items():self.assertEqual(m.sha(p),h)
            for file in ['CurrentBest_097755_control.csv','I2_ViTL336_standalone.csv','F0_B336_L336_equal_backbone_submission.csv']:
                frame=pd.read_csv(output/file,dtype={'image_id':str});self.assertEqual(frame.image_id.tolist(),ids);self.assertTrue(frame.label.between(0,19).all())
            self.assertTrue(pd.read_csv(output/'CurrentBest_097755_control.csv').equals(pd.read_csv(i1/'F0_I1_336_equal_branch_submission.csv')))
            self.assertFalse(result['submitted']);self.assertTrue(result['original_scored_predictions_reproduced']);self.assertEqual(result['large_batch_size'],2)
        n=nbformat.read(Path(m.__file__).with_name('I2_ViTL336_Ensemble.ipynb'),as_version=4);nbformat.validate(n)
        for c in n.cells:
            if c.cell_type=='code' and not c.source.startswith('%'):compile(c.source,'notebook','exec')
        scope={'__name__':'large_notebook_test'}
        exec(compile(n.cells[5].source,'embedded_implementation','exec'),scope)
        self.assertEqual(scope['SETTINGS'],m.SETTINGS)
        self.assertEqual(scope['build_head'](1024)[0].in_features,1024)
        self.assertTrue(torch.equal(scope['blend_large'](torch.eye(20)[:2],torch.eye(20)[:2]),torch.eye(20)[:2]))

if __name__=='__main__':unittest.main()
