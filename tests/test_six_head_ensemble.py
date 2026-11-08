import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import nbformat
import six_head_ensemble as m

class Encoder(nn.Module):
    def __init__(self,dim):super().__init__();self.embed_dim=dim;self.anchor=nn.Parameter(torch.zeros(1))
    def forward(self,x):return torch.zeros(len(x),self.embed_dim,device=x.device)+self.anchor

class SixHeadTests(unittest.TestCase):
    def test_fixed_weights(self):
        s=torch.zeros(3,2,20);b=torch.zeros(3,2,20);s[:,:,0]=1;b[:,:,1]=1
        blended=m.average_branches(s,b)
        self.assertTrue(torch.equal(blended[:,:2],torch.full((2,2),.5)))
        with self.assertRaises(AssertionError):m.average_branches(s[:2],b)
    def test_end_to_end_preserves_sources_and_trains_only_missing(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f0=root/'f0';f0.mkdir();ref=root/'reference';ref.mkdir()
            image=root/'image.png';Image.new('RGB',(32,32),(100,110,120)).save(image)
            mapping={f'{i:03d}_car':i for i in range(20)};ids=['009','001']
            targets=np.tile(np.arange(20),10);train_targets=np.tile(np.arange(20),35)
            meta={'backbone':'dinov2_vitb14','seed_aug':123,'n_aug':4,'include_original':True,'layout':'view-major','feature_dim':768,'test_ids':ids,'weights_sha256':m.weight_digest(Encoder(768)),
                  'manifest':[[],[],[{'path':'image.png','sha256':m.sha(image)}]]}
            features={'metadata':meta,'X_train':torch.zeros(3500,768),'X_val':torch.zeros(200,768),'X_test':torch.zeros(2,768),
                      'y_train':torch.tensor(train_targets).repeat(5),'y_val':torch.tensor(targets)}
            signature=m.hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            cache=root/'cache.pt';torch.save(features,cache)
            small_probs=[];records=[];old_paths=[]
            for seed in m.SEEDS:
                torch.manual_seed(seed);head=m.build_head(384).eval()
                c={'backbone':'dinov2_vits14','seed_aug':123,'seed_head':seed,'class_to_idx':mapping,
                   'feature_mean':torch.zeros(384),'feature_std':torch.ones(384),
                   'head_config':{'hidden_dim':128,'use_bn':True,'dropout':.3},'head_state_dict':head.state_dict()}
                path=f0/f'seed{seed}.pt';torch.save(c,path);old_paths.append(path)
                with torch.inference_mode():p=head(torch.zeros(200,384)).softmax(1)
                small_probs.append(p);records.append({'seed':seed,'checkpoint':str(path),**m.metrics(targets,p)})
            fs={'head_seeds':m.SEEDS,'augmentation_seed':123,'hub_ref':m.HUB_REF,'weights':[1/3]*3,
                'weights_sha256':m.weight_digest(Encoder(384)),'individual_validation':records,'ensemble_validation':m.metrics(targets,torch.stack(small_probs).mean(0))}
            (f0/'ensemble_summary.json').write_text(json.dumps(fs))
            head=m.build_head(768).eval();mean=features['X_train'].mean(0);std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
            with torch.inference_mode():bp=head(torch.zeros(200,768)).softmax(1)
            base={'backbone':'dinov2_vitb14','seed_aug':123,'seed_head':2026,'class_to_idx':mapping,'hub_ref':m.HUB_REF,
                  'head_state_dict':head.state_dict(),'head_config':{'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},
                  'feature_mean':mean,'feature_std':std,'feature_metadata':meta,'feature_signature':signature,
                  'metrics':m.metrics(targets,bp),'training_config':m.SETTINGS}
            base_path=root/'base.pt';torch.save(base,base_path);old_paths.append(base_path)
            old_prob=.5*torch.stack(small_probs).mean(0)+.5*bp
            rs={'checkpoint_sha256':{str(p):m.sha(p) for p in old_paths},'validation':{'equal_branch_ensemble':m.metrics(targets,old_prob)}}
            (ref/'summary.json').write_text(json.dumps(rs))
            m.export(ref/'F0_I0_equal_branch_ensemble_submission.csv',ids,old_prob[:2])
            hashes={p:m.sha(p) for p in [*old_paths,cache]}
            fake_train=type('Train',(),{'class_to_idx':mapping,'targets':train_targets.tolist(),'__len__':lambda self:700})()
            data=(mapping,[image]*200,targets,[image]*2,ids)
            with patch.object(m,'load_data',return_value=data),patch.object(m.datasets,'ImageFolder',return_value=fake_train),patch.object(m,'pinned_backbone',side_effect=lambda name,expected,dim,device:Encoder(dim).requires_grad_(False).eval()):
                output,result=m.run_six(root,f0,base_path,ref,cache,root,batch_size=16)
            self.assertEqual(result['new_head_seeds'],[42,123]);self.assertFalse(result['original_seed2026_retrained'])
            self.assertFalse(result['submitted']);self.assertTrue(result['original_scored_csv_reproduced'])
            self.assertEqual(len(list(output.glob('*.pt'))),2)
            for path,value in hashes.items():self.assertEqual(m.sha(path),value)
            frame=pd.read_csv(output/'SixHead_equal_branch_ensemble_submission.csv',dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),ids);self.assertTrue(frame.label.between(0,19).all())
            for seed in (42,123):
                saved=torch.load(output/f'I0_dinov2_vitb14_p3_e4_aug123_head{seed}.pt',weights_only=True)
                self.assertEqual(saved['seed_head'],seed);self.assertTrue(torch.equal(saved['feature_mean'],mean))
            with self.assertRaises(AssertionError):m.train_missing_head(2026,base,features,output,'cpu')
            with patch.object(m.datasets,'ImageFolder',return_value=fake_train):
                with self.assertRaises(AssertionError):m.feature_cache(base,cache,root,mapping,targets,['WRONG'])
        n=nbformat.read(Path(m.__file__).with_name('F0_I0_SixHead_Ensemble.ipynb'),as_version=4);nbformat.validate(n)
        for c in n.cells:
            if c.cell_type=='code' and not c.source.startswith('%'):compile(c.source,'notebook','exec')

if __name__=='__main__':unittest.main()
