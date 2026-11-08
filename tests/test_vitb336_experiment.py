import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import nbformat
import vitb336_experiment as m

class Encoder(nn.Module):
    def __init__(self,dim):super().__init__();self.embed_dim=dim;self.anchor=nn.Parameter(torch.zeros(1));self.calls=0;self.sizes=[]
    def forward(self,x):
        self.calls+=1;self.sizes.append(x.shape[-1]);return torch.zeros(len(x),self.embed_dim,device=x.device)+self.anchor

class B336Tests(unittest.TestCase):
    def test_resolution_and_weights(self):
        evaluation,augmentation=m.transforms336()
        image=Image.new('RGB',(400,500),(100,120,130))
        self.assertEqual(evaluation(image).shape,(3,336,336));self.assertEqual(augmentation(image).shape,(3,336,336))
        s=torch.zeros(3,2,20);s[:,:,0]=1;b=torch.zeros(2,20);b[:,1]=1
        self.assertTrue(torch.equal(m.blend336(s,b)[:,:2],torch.full((2,2),.5)))
        with self.assertRaises(AssertionError):m.blend336(s[:2],b)
    def test_end_to_end_cache_normalization_and_control_preserved(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f0=root/'f0';f0.mkdir();ref=root/'reference';ref.mkdir()
            image=root/'image.png';Image.new('RGB',(32,32),(100,110,120)).save(image)
            mapping={f'{i:03d}_car':i for i in range(20)};ids=['009','001'];targets=np.tile(np.arange(20),10)
            train_targets=np.tile(np.arange(20),35)
            model_s=Encoder(384).eval().requires_grad_(False);model_b=Encoder(768).eval().requires_grad_(False)
            small_prob=[];records=[];old_paths=[]
            for seed in m.SEEDS:
                torch.manual_seed(seed);head=m.build_head(384).eval()
                checkpoint={'backbone':'dinov2_vits14','seed_aug':123,'seed_head':seed,'class_to_idx':mapping,
                    'feature_mean':torch.zeros(384),'feature_std':torch.ones(384),'head_state_dict':head.state_dict(),
                    'head_config':{'hidden_dim':128,'use_bn':True,'dropout':.3}}
                path=f0/f'seed{seed}.pt';torch.save(checkpoint,path);old_paths.append(path)
                with torch.inference_mode():p=head(torch.zeros(200,384)).softmax(1)
                small_prob.append(p);records.append({'seed':seed,'checkpoint':str(path),**m.metrics(targets,p)})
            fs={'head_seeds':m.SEEDS,'augmentation_seed':123,'hub_ref':m.HUB_REF,'weights':[1/3]*3,'individual_validation':records,
                'weights_sha256':m.weight_digest(model_s),'ensemble_validation':m.metrics(targets,torch.stack(small_prob).mean(0))}
            (f0/'ensemble_summary.json').write_text(json.dumps(fs))
            head=m.build_head(768).eval()
            with torch.inference_mode():bp=head(torch.zeros(200,768)).softmax(1)
            meta={'backbone':'dinov2_vitb14','feature_dim':768,'seed_aug':123,'n_aug':4,'include_original':True,
                'class_to_idx':mapping,'test_ids':ids,'weights_sha256':m.weight_digest(model_b),
                'manifest':[[],[],[{'path':'image.png','sha256':m.sha(image)}]]}
            base={'backbone':'dinov2_vitb14','hub_ref':m.HUB_REF,'seed_aug':123,'seed_head':2026,'class_to_idx':mapping,
                'head_config':{'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},
                'head_state_dict':head.state_dict(),'feature_mean':torch.zeros(768),'feature_std':torch.ones(768),
                'feature_metadata':meta,'training_config':m.SETTINGS,'metrics':m.metrics(targets,bp)}
            base_path=root/'base.pt';torch.save(base,base_path);old_paths.append(base_path)
            previous=.5*torch.stack(small_prob).mean(0)+.5*bp
            original_hashes={p:m.sha(p) for p in old_paths}
            rs={'checkpoint_sha256':{str(p):m.sha(p) for p in old_paths},'validation':{'equal_branch_ensemble':m.metrics(targets,previous)}}
            (ref/'summary.json').write_text(json.dumps(rs));m.export(ref/'F0_I0_equal_branch_ensemble_submission.csv',ids,previous[:2])
            fake_train=type('Train',(),{'class_to_idx':mapping,'targets':train_targets.tolist(),'samples':[(str(image),int(y)) for y in train_targets],'__len__':lambda self:700})()
            data=(mapping,[image]*200,targets,[image]*2,ids)
            with patch.object(m,'load_data',return_value=data),patch.object(m.datasets,'ImageFolder',return_value=fake_train),patch.object(m,'pinned_backbone',side_effect=[model_s,model_b]):
                output,result=m.run336(root,f0,base_path,ref,root,batch_size=8)
                calls=model_b.calls
                features,key,cache=m.get_features336(model_b,base,root,data[1],targets,data[3],ids,mapping,output.parent/'cache_336','cpu',8)
                self.assertEqual(model_b.calls,calls)
                checkpoint=torch.load(result['new_checkpoint'],map_location='cpu',weights_only=True)
                self.assertEqual(checkpoint['resolution'],336)
                self.assertTrue(torch.equal(checkpoint['feature_mean'],features['X_train'].mean(0)))
                self.assertTrue(torch.equal(checkpoint['feature_std'],features['X_train'].std(0,unbiased=False).clamp_min(1e-6)))
                self.assertEqual(checkpoint['feature_metadata']['image_size'],336)
                self.assertEqual(features['X_train'].shape,(3500,768))
                self.assertTrue(torch.equal(features['y_train'],torch.tensor(train_targets).repeat(5)))
                bad=dict(features);bad['X_train']=torch.zeros(3500,384);torch.save(bad,cache)
                with patch('vitb336_experiment.Images',side_effect=RuntimeError('regeneration attempted')):
                    with self.assertRaisesRegex(RuntimeError,'regeneration attempted'):
                        m.get_features336(model_b,base,root,data[1],targets,data[3],ids,mapping,output.parent/'cache_336','cpu',8)
            self.assertIn(224,model_b.sizes);self.assertIn(336,model_b.sizes)
            for p,h in original_hashes.items():self.assertEqual(m.sha(p),h)
            self.assertTrue(result['original_scored_predictions_reproduced']);self.assertFalse(result['submitted'])
            for filename in ['CurrentBest_097551_control.csv','I1_ViTB336_standalone.csv','F0_I1_336_equal_branch_submission.csv']:
                frame=pd.read_csv(output/filename,dtype={'image_id':str})
                self.assertEqual(frame.image_id.tolist(),ids);self.assertTrue(frame.label.between(0,19).all())
            control=pd.read_csv(output/'CurrentBest_097551_control.csv')
            self.assertTrue(control.equals(pd.read_csv(ref/'F0_I0_equal_branch_ensemble_submission.csv')))
        notebook=nbformat.read(Path(m.__file__).with_name('I1_ViTB336_Ensemble.ipynb'),as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'notebook','exec')

if __name__=='__main__':unittest.main()
