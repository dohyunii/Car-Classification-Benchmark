"""J3 predeclared selector/export tests using synthetic probabilities and models."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

import select_pooled_candidate as selector


class TinyBackbone(nn.Module):
    num_prefix_tokens=1
    def __init__(self):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward_features(self,images):
        x=images.mean(dim=(2,3)).repeat(1,128)+self.anchor
        return torch.stack([x,x+.1,x+.3],dim=1)


class SelectPooledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.run=self.root/'J3'
        (self.run/'metrics').mkdir(parents=True)
        (self.run/'checkpoints').mkdir()
        self.baseline=self.root/'J0'
        self.baseline.mkdir()
        (self.baseline/'summary.json').write_text(json.dumps({'selection':{'accuracy':.975,'head_seed':2026,'resolution':224}}))
        self.labels=np.repeat(np.arange(20),10)
        self.paths=np.asarray([f'val/{i:03d}.jpg' for i in range(200)])
        self.classes={f'{i:03d}_car':i for i in range(20)}
        self.class_names=np.asarray(list(self.classes))
        self.model=TinyBackbone().eval()
        self.data_root=self.root/'data'
        (self.data_root/'test').mkdir(parents=True)
        for name in ('0001','0002'):
            Image.new('RGB',(40,50),(5,10,20)).save(self.data_root/'test'/f'{name}.png')
        (self.data_root/'sample_submission.csv').write_text('image_id,label\n0002,0\n0001,0\n')
        self.data={'paths':[[],[Path(p) for p in self.paths],
                            [self.data_root/'test'/'0002.png',self.data_root/'test'/'0001.png']],
                   'ids':['0002','0001'],'manifest':[{'synthetic':True}],
                   'labels':[torch.tensor([]),torch.from_numpy(self.labels)],'class_to_idx':self.classes}
        self.records=[]

    def tearDown(self): self.temp.cleanup()

    def setup_probabilities(self,wrong_by_seed):
        self.records=[]
        for seed in (42,123,2026):
            predicted=self.labels.copy()
            wrong=np.asarray(wrong_by_seed[seed],dtype=int)
            predicted[wrong]=(predicted[wrong]+1)%20
            p=np.full((200,20),.01/19,dtype=np.float32)
            p[np.arange(200),predicted]=.99
            checkpoint=self.run/'checkpoints'/f'J3_head{seed}.pt'
            record={'head_seed':seed,'accuracy':float((predicted==self.labels).mean()),
                    'checkpoint':str(checkpoint),'augmentation_seed':123,'resolution':224,
                    'feature_dim':768,'pooling':'concat(final_normalized_cls,mean(final_normalized_patch_tokens))'}
            self.records.append(record)
            np.savez_compressed(self.run/'metrics'/f'{checkpoint.stem}_validation_probabilities.npz',
                  target=self.labels,path=self.paths,class_names=self.class_names,probabilities=p)
            head=nn.Linear(768,20)
            with torch.no_grad(): head.weight.zero_();head.bias.zero_();head.bias[3]=10
            torch.save({'record':record,'head_state_dict':head.state_dict(),
                        'feature_mean':torch.zeros(768),'feature_std':torch.ones(768),
                        'class_to_idx':self.classes,'feature_metadata':{'manifest':self.data['manifest'],
                         'weight_sha256':selector.common.model_fingerprint(self.model)}},checkpoint)
        (self.run/'J3_summary.json').write_text(json.dumps({'records':self.records}))

    def select(self):
        return selector.select_and_export(self.run,self.data_root,self.baseline)

    def test_no_gain_and_baseline_ties_do_not_infer_or_load_test_data(self):
        self.setup_probabilities({42:range(195,200),123:range(195,200),2026:range(195,200)})
        with patch.object(selector.common,'load_data',side_effect=AssertionError('no test/data load')), \
             patch.object(selector.timm,'create_model',side_effect=AssertionError('no model load')):
            decision=self.select()
        self.assertEqual(decision['selection'],'baseline')
        self.assertNotIn('submission',decision)
        self.assertFalse(decision['submitted'])
        self.assertFalse((self.run/'submissions').exists())
        self.assertEqual(decision['pooled_equal_ensemble']['seeds'],[42,123,2026])
        self.assertEqual(decision['pooled_equal_ensemble']['weights'],[1/3]*3)
        # Improving over a weaker single head but merely tying baseline is insufficient.
        self.setup_probabilities({42:range(195,200),123:range(195,200),2026:range(194,200)})
        with patch.object(selector.common,'load_data',side_effect=AssertionError('no data load')):
            tied=self.select()
        self.assertEqual(tied['selection'],'baseline')

    def test_single_strict_gain_and_ensemble_ties_keep_fewer_heads_export_order(self):
        # All three get196 right: ensemble ties the fixed2026 head, so select one.
        self.setup_probabilities({seed:range(196,200) for seed in (42,123,2026)})
        with patch.object(selector.common,'load_data',return_value=self.data), \
             patch.object(selector.timm,'create_model',return_value=self.model) as create, \
             patch.object(selector.j3,'build_head',side_effect=lambda:nn.Linear(768,20)) as build:
            decision=self.select()
        self.assertEqual(decision['selection'],'single_2026')
        self.assertEqual(len(decision['checkpoints']),1)
        self.assertIn('2026',decision['checkpoints'][0])
        self.assertFalse(decision['submitted'])
        create.assert_called_once()
        self.assertEqual(build.call_count,1)
        csv=Path(decision['submission'])
        content=csv.read_bytes()
        submission=pd.read_csv(csv,dtype={'image_id':str})
        self.assertEqual(submission.image_id.tolist(),['0002','0001'])
        self.assertEqual(list(submission.columns),['image_id','label'])
        self.assertEqual(submission.label.tolist(),[3,3])
        with patch.object(selector.common,'load_data',return_value=self.data), \
             patch.object(selector.timm,'create_model',return_value=self.model), \
             patch.object(selector.j3,'build_head',side_effect=lambda:nn.Linear(768,20)):
            with self.assertRaises(FileExistsError): self.select()
        self.assertEqual(csv.read_bytes(),content)

    def test_all_three_equal_average_requires_strict_gain_over_single_and_baseline(self):
        # Disjoint errors: each head195/200; fixed all-three mean corrects them.
        self.setup_probabilities({42:range(0,5),123:range(5,10),2026:range(10,15)})
        with patch.object(selector.common,'load_data',return_value=self.data), \
             patch.object(selector.timm,'create_model',return_value=self.model), \
             patch.object(selector.j3,'build_head',side_effect=lambda:nn.Linear(768,20)) as build:
            decision=self.select()
        self.assertEqual(decision['selection'],'equal_3seed_ensemble')
        self.assertEqual(decision['pooled_equal_ensemble']['accuracy'],1.)
        self.assertEqual(build.call_count,3)
        self.assertEqual(len(decision['checkpoints']),3)

    def test_misaligned_targets_paths_probabilities_or_seeds_are_rejected(self):
        wrong={seed:range(195,200) for seed in (42,123,2026)}
        for field in ('target','path','class_names','probabilities','negative_probability'):
            self.setup_probabilities(wrong)
            path=self.run/'metrics'/'J3_head42_validation_probabilities.npz'
            loaded=dict(np.load(path,allow_pickle=False))
            if field=='target': loaded[field]=loaded[field][::-1]
            elif field=='path': loaded[field]=loaded[field][::-1]
            elif field=='class_names': loaded[field]=loaded[field][::-1]
            elif field=='probabilities': loaded[field][0,0]=np.nan
            else:
                p=loaded['probabilities']
                correction=p[0,5]+.1
                p[0,5]=-.1;p[0,6]+=correction
            np.savez_compressed(path,**loaded)
            with self.assertRaises(AssertionError): self.select()
        self.setup_probabilities(wrong)
        records=json.loads((self.run/'J3_summary.json').read_text())
        records['records'][0]['head_seed']=123
        (self.run/'J3_summary.json').write_text(json.dumps(records))
        with self.assertRaises(AssertionError): self.select()

    def test_gain_export_rejects_dataset_order_and_invalid_saved_statistics(self):
        self.setup_probabilities({seed:range(196,200) for seed in (42,123,2026)})
        shuffled=dict(self.data,paths=[[],list(reversed(self.data['paths'][1])),self.data['paths'][2]])
        with patch.object(selector.common,'load_data',return_value=shuffled), \
             patch.object(selector.timm,'create_model',side_effect=AssertionError('must reject before model load')):
            with self.assertRaisesRegex(AssertionError,'path order'): self.select()
        for statistic,value in [('feature_std',0.),('feature_mean',float('nan'))]:
            self.setup_probabilities({seed:range(196,200) for seed in (42,123,2026)})
            checkpoint=Path(self.records[-1]['checkpoint'])
            saved=torch.load(checkpoint,weights_only=True)
            saved[statistic][0]=value
            torch.save(saved,checkpoint)
            with patch.object(selector.common,'load_data',return_value=self.data), \
                 patch.object(selector.timm,'create_model',return_value=self.model), \
                 patch.object(selector.j3,'build_head',side_effect=lambda:nn.Linear(768,20)):
                with self.assertRaises(AssertionError): self.select()
        self.assertFalse((self.run/'submissions').exists())


if __name__=='__main__': unittest.main()
