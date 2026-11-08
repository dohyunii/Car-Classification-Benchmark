import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import nbformat
import cross_backbone_ensemble as m

class Encoder(nn.Module):
    def __init__(self,dim):
        super().__init__();self.embed_dim=dim;self.anchor=nn.Parameter(torch.zeros(1))
    def forward(self,x):return torch.zeros(len(x),self.embed_dim,device=x.device)+self.anchor

class CrossEnsembleTests(unittest.TestCase):
    def test_branch_weights_and_gates(self):
        s=torch.zeros(3,2,20);s[:,:,0]=1
        b=torch.zeros(2,20);b[:,1]=1
        small,blend=m.blend_probabilities(s,b)
        self.assertTrue(torch.equal(blend[:,0],torch.tensor([.5,.5])))
        self.assertTrue(torch.equal(blend[:,1],torch.tensor([.5,.5])))
        with self.assertRaises(AssertionError):m.blend_probabilities(s[:2],b)
        with self.assertRaises(AssertionError):m.reproduce({'accuracy':.98,'correct':196},{'accuracy':.975,'correct':195},'source')
    def test_checkpoint_inference_export_and_notebook(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f0=root/'f0';f0.mkdir()
            image=root/'image.png';Image.new('RGB',(32,32),(100,110,120)).save(image)
            mapping={f'{i:03d}_car':i for i in range(20)};ids=['009','001']
            targets=np.zeros(200,dtype=int)
            small=[];records=[]
            for seed in m.SEEDS:
                torch.manual_seed(seed);head=m.build_head(384).eval()
                c={'backbone':'dinov2_vits14','seed_aug':123,'seed_head':seed,'class_to_idx':mapping,
                   'feature_mean':torch.zeros(384),'feature_std':torch.ones(384),
                   'head_config':{'hidden_dim':128,'use_bn':True,'dropout':.3},'head_state_dict':head.state_dict()}
                path=f0/f'seed{seed}.pt';torch.save(c,path)
                with torch.inference_mode():p=head(torch.zeros(200,384)).softmax(1)
                small.append(p);records.append({'seed':seed,'checkpoint':str(path),**m.metrics(targets,p)})
            original=m.metrics(targets,torch.stack(small).mean(0))
            summary={'head_seeds':m.SEEDS,'augmentation_seed':123,'weights':[1/3]*3,'backbone':'dinov2_vits14',
                     'hub_ref':m.HUB_REF,'individual_validation':records,'ensemble_validation':original,
                     'weights_sha256':m.weight_digest(Encoder(384))}
            (f0/'ensemble_summary.json').write_text(json.dumps(summary))
            head=m.build_head(768).eval();mean=torch.ones(768)*.2;std=torch.ones(768)*2
            with torch.inference_mode():p=head((torch.zeros(200,768)-mean)/std).softmax(1)
            base={'backbone':'dinov2_vitb14','seed_aug':123,'seed_head':2026,'class_to_idx':mapping,
                  'feature_mean':mean,'feature_std':std,'head_state_dict':head.state_dict(),'hub_ref':m.HUB_REF,
                  'head_config':{'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},
                  'metrics':m.metrics(targets,p),'feature_metadata':{'backbone':'dinov2_vitb14',
                  'weights_sha256':m.weight_digest(Encoder(768)),'test_ids':ids,
                  'manifest':[[],[],[{'path':'image.png','sha256':m.sha(image)}]]}}
            self.assertNotIn('model',base['feature_metadata'])
            self.assertEqual(base['feature_metadata']['backbone'],'dinov2_vitb14')
            bp=root/'base.pt';torch.save(base,bp)
            fake_data=(mapping,[image]*200,targets,[image]*2,ids)
            with patch.object(m,'load_data',return_value=fake_data),patch('torch.hub.load',side_effect=[Encoder(384),Encoder(768)]) as hub:
                output,result=m.run(root,f0,bp,root,batch_size=16)
            self.assertEqual(hub.call_count,2);self.assertFalse(result['submitted']);self.assertFalse(result['trained'])
            self.assertEqual(result['effective_head_weights'],[1/6,1/6,1/6,1/2])
            frame=pd.read_csv(output/'F0_I0_equal_branch_ensemble_submission.csv',dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),ids);self.assertTrue(frame.label.between(0,19).all())
            self.assertEqual(list(frame.columns),['image_id','label'])
            with self.assertRaises(FileExistsError):m.export(output/'F0_I0_equal_branch_ensemble_submission.csv',ids,p[:2])
            image.write_bytes(image.read_bytes()+b'changed')
            with patch.object(m,'load_data',return_value=fake_data),patch('torch.hub.load') as hub:
                with self.assertRaises(AssertionError):m.run(root,f0,bp,root)
                hub.assert_not_called()
        notebook=nbformat.read(Path(m.__file__).with_name('F0_I0_CrossBackbone_Ensemble.ipynb'),as_version=4)
        nbformat.validate(notebook)
        embedded=notebook.cells[5].source
        self.assertEqual(embedded,Path(m.__file__).read_text().split("if __name__=='__main__':")[0])
        for cell in notebook.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'notebook','exec')

if __name__=='__main__':unittest.main()
