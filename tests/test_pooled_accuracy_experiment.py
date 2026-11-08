"""Paired J3 representation tests with synthetic images/tokens; no real validation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from PIL import Image

import pooled_accuracy_experiment as j3


class TinyTokenEncoder(nn.Module):
    num_prefix_tokens = 1
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1),requires_grad=False)
        self.pretrained_cfg = {'source':'synthetic'}
        self.calls = 0
    def forward_features(self,images):
        self.calls += 1
        cls = images[:,:,::14,::14].flatten(1)[:,:384] + self.anchor
        tokens = [cls]
        if self.num_prefix_tokens == 2:
            tokens.append(torch.full_like(cls,999))
        return torch.stack(tokens+[cls+.1,cls+.3],dim=1)
    def forward(self,images):
        return self.forward_features(images)[:,0]


class PooledAccuracyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root/'output'
        for directory in ('checkpoints','metrics','cache'):(self.output/directory).mkdir(parents=True)
        data_root=self.root/'data'
        rng=np.random.default_rng(42)
        for split in ('train','val'):
            for label in range(20):
                folder=data_root/split/f'{label:03d}_car'
                folder.mkdir(parents=True)
                Image.fromarray(rng.integers(0,256,(70,80,3),dtype=np.uint8)).save(folder/'sample.png')
        (data_root/'test').mkdir()
        for name in ('a','b'):
            Image.fromarray(rng.integers(0,256,(70,80,3),dtype=np.uint8)).save(data_root/'test'/f'{name}.png')
        (data_root/'sample_submission.csv').write_text('image_id,label\nb,0\na,0\n')
        self.data=j3.common.load_data(data_root,(20,20,2))
        self.model=TinyTokenEncoder().eval()
        self.reference,self.source_key=j3.j0.get_features(self.model,self.data,224,self.output,'cpu',batch_size=8)

    def tearDown(self): self.temp.cleanup()

    def features(self,**kwargs):
        return j3.get_features(self.model,self.data,self.reference,self.source_key,self.output,'cpu',8,**kwargs)

    def test_mean_patch_pooling_excludes_all_prefix_tokens(self):
        images=torch.rand(2,3,224,224)
        for prefix in (1,2):
            self.model.num_prefix_tokens=prefix
            result=j3.pooled_forward(self.model,images)
            self.assertEqual(result.shape,(2,768))
            self.assertTrue(torch.allclose(result[:,384:],result[:,:384]+.2,rtol=0,atol=1e-6))

    def test_exact_paired_prefix_cache_alignment_and_no_bypass(self):
        payload,key=self.features()
        self.assertEqual(payload['X_train'].shape,(100,768))
        self.assertTrue(torch.equal(payload['X_train'][:,:384],self.reference['X_train']))
        self.assertTrue(torch.equal(payload['X_val'][:,:384],self.reference['X_val']))
        self.assertTrue(torch.equal(payload['y_train'],self.data['labels'][0].repeat(5)))
        self.assertFalse(torch.equal(payload['X_train'][20:40],payload['X_train'][40:60]))
        with patch.object(self.model,'forward_features',side_effect=AssertionError('cache must hit')):
            cached,same=self.features()
        self.assertEqual(key,same)
        self.assertTrue(torch.equal(cached['X_train'],payload['X_train']))
        altered=dict(self.reference,X_train=self.reference['X_train'].clone())
        altered['X_train'][0,0]+=1
        with self.assertRaisesRegex(AssertionError,'CLS prefix differs'):
            j3.get_features(self.model,self.data,altered,self.source_key,self.output,'cpu',8)
        with self.assertRaisesRegex(AssertionError,'batch size'):
            j3.get_features(self.model,self.data,self.reference,self.source_key,self.output,'cpu',4)
        with torch.no_grad(): self.model.anchor.add_(1)
        with self.assertRaisesRegex(AssertionError,'weights differ'):
            self.features()

    def test_fresh_head_seeds_training_statistics_and_checkpoint_roundtrip(self):
        features,key=self.features()
        mean=features['X_train'].mean(0)
        std=features['X_train'].std(0,unbiased=False).clamp_min(1e-6)
        features['X_val']=features['X_val']+1000
        with patch.dict(j3.TRAIN_CONFIG,epochs=2,patience=1):
            first=j3.train_head(features,42,key,self.data,self.output,'cpu')
            second=j3.train_head(features,2026,key,self.data,self.output,'cpu')
        saved=torch.load(first['checkpoint'],map_location='cpu',weights_only=True)
        other=torch.load(second['checkpoint'],map_location='cpu',weights_only=True)
        self.assertTrue(torch.equal(saved['feature_mean'],mean))
        self.assertTrue(torch.equal(saved['feature_std'],std))
        self.assertEqual(saved['head_state_dict']['0.weight'].shape,(128,768))
        self.assertEqual(saved['record']['head_seed'],42)
        self.assertEqual(saved['record']['augmentation_seed'],123)
        self.assertEqual(saved['class_to_idx'],self.data['class_to_idx'])
        self.assertFalse(torch.equal(saved['head_state_dict']['0.weight'],other['head_state_dict']['0.weight']))
        probability_path=self.output/'metrics'/(Path(first['checkpoint']).stem+'_validation_probabilities.npz')
        probabilities=np.load(probability_path)
        self.assertEqual(probabilities['probabilities'].shape,(20,20))
        np.testing.assert_allclose(probabilities['probabilities'].sum(1),1,rtol=0,atol=1e-6)
        np.testing.assert_allclose(probabilities['probabilities'],torch.tensor(probabilities['logits']).softmax(1).numpy(),rtol=0,atol=1e-7)
        self.assertEqual(probabilities['path'].tolist(),[str(p) for p in self.data['paths'][1]])
        np.testing.assert_array_equal(probabilities['target'],features['y_val'].numpy())
        self.assertEqual(probabilities['class_names'].tolist(),list(self.data['class_to_idx']))
        head=j3.build_head();head.load_state_dict(saved['head_state_dict'],strict=True)
        with self.assertRaises(FileExistsError):
            with patch.dict(j3.TRAIN_CONFIG,epochs=1):
                j3.train_head(features,42,key,self.data,self.output,'cpu')


if __name__=='__main__': unittest.main()
