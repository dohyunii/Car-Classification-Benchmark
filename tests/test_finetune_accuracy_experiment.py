"""J2 synthetic gradient/freeze/gate checks; never loads actual pretrained models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import finetune_accuracy_experiment as j2


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3,384)
        self.blocks = nn.ModuleList([nn.Linear(384,384) for _ in range(12)])
        self.norm = nn.LayerNorm(384)
    def forward(self, images):
        x = self.stem(images.mean(dim=(2,3)))
        for block in self.blocks:
            x = x + block(x).tanh() * 0.1
        return self.norm(x)


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        j2.common.set_seed(2026)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backbone = TinyBackbone()
        self.head = j2.j0.build_head()
        self.source = {'head_state_dict':{k:v.clone() for k,v in self.head.state_dict().items()},
                       'feature_mean':torch.zeros(384),'feature_std':torch.ones(384),
                       'feature_metadata':{'weight_sha256':j2.common.model_fingerprint(self.backbone)},
                       'record':{'accuracy':0.,'correct':0}}
        self.model = j2.build_model(self.source,'cpu',backbone=self.backbone)
        images = torch.randn(16,3,8,8)
        labels = torch.arange(16) % 20
        self.loader = DataLoader(TensorDataset(images,labels),batch_size=8,shuffle=False)
        baseline,_,_ = j2.evaluate(self.model,self.loader,'cpu')
        self.source['record'].update(accuracy=baseline['accuracy'],correct=baseline['correct'])
        self.source_path = self.root/'source.pt'
        torch.save(self.source,self.source_path)
        self.data = {'paths':[[],[Path(f'val/{i}.jpg') for i in range(16)],[]],
                     'labels':[torch.tensor([]),labels],
                     'class_to_idx':{f'{i:03d}_car':i for i in range(20)}}

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_trainable_parameters_eval_modes_and_gradient_flow(self):
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(self.model.head.training)
        self.assertFalse(self.model.head[1].training)
        self.assertTrue(self.model.head[3].training)
        for name, parameter in self.model.backbone.named_parameters():
            expected = name.startswith(('blocks.10.','blocks.11.','norm.'))
            self.assertEqual(parameter.requires_grad,expected,name)
        early_outputs=[]
        hook=self.model.backbone.blocks[9].register_forward_hook(lambda m,i,o:early_outputs.append(o.requires_grad))
        original_bn={k:v.clone() for k,v in self.model.head[1].state_dict().items() if 'running_' in k or 'num_batches' in k}
        x,y = next(iter(self.loader))
        nn.functional.cross_entropy(self.model(x),y).backward()
        hook.remove()
        self.assertEqual(early_outputs,[False])
        self.assertIsNone(self.model.backbone.blocks[0].weight.grad)
        self.assertGreater(self.model.backbone.blocks[10].weight.grad.abs().sum().item(),0)
        self.assertGreater(self.model.backbone.blocks[11].weight.grad.abs().sum().item(),0)
        self.assertGreater(self.model.backbone.norm.weight.grad.abs().sum().item(),0)
        self.assertGreater(self.model.head[1].weight.grad.abs().sum().item(),0)
        for name,value in original_bn.items():
            self.assertTrue(torch.equal(value,self.model.head[1].state_dict()[name]))

    def test_baseline_and_weight_mismatch_gates(self):
        measured,_,_ = j2.verify_baseline(self.model,self.loader,self.source,'cpu')
        self.assertEqual(measured['correct'],self.source['record']['correct'])
        invalid=dict(self.source,record=dict(self.source['record'],correct=self.source['record']['correct']+1))
        with self.assertRaises(AssertionError): j2.verify_baseline(self.model,self.loader,invalid,'cpu')
        before = j2.cpu_state(self.model)
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer must not be created')) as optimizer:
            with self.assertRaisesRegex(AssertionError,'epoch-zero gate failed'):
                j2.train_finetune(self.model,self.loader,self.loader,invalid,self.data,self.source_path,
                                  self.root/'failed-gate','cpu',epochs=1)
        optimizer.assert_not_called()
        for name,value in before.items():
            self.assertTrue(torch.equal(value,self.model.state_dict()[name]))
        changed = TinyBackbone()
        with self.assertRaises(AssertionError): j2.build_model(self.source,'cpu',backbone=changed)

    def test_epoch_zero_retained_strict_ties_and_frozen_state_restored(self):
        original = j2.cpu_state(self.model)
        bn_before={k:v.clone() for k,v in self.model.head[1].state_dict().items() if 'running_' in k or 'num_batches' in k}
        measured,predicted,probabilities = j2.evaluate(self.model,self.loader,'cpu')
        observed_train_updates=[]
        # Higher F1 at identical accuracy must not qualify or reset patience.
        calls=[0]
        def evaluation(model,loader,device):
            calls[0]+=1
            current=j2.cpu_state(model)
            if calls[0]>1:
                observed_train_updates.append(not torch.equal(current['backbone.blocks.10.weight'],original['backbone.blocks.10.weight']))
                for name,parameter in model.backbone.named_parameters():
                    if not parameter.requires_grad:
                        self.assertTrue(torch.equal(parameter,original['backbone.'+name]))
                for name,value in bn_before.items():
                    self.assertTrue(torch.equal(value,model.head[1].state_dict()[name]))
            metrics=dict(measured,macro_f1=min(1.0,measured['macro_f1']+0.01*calls[0]))
            model.eval()
            return metrics,predicted,probabilities
        output=self.root/'J2'
        with patch.object(j2,'evaluate',side_effect=evaluation):
            result=j2.train_finetune(self.model,self.loader,self.loader,self.source,self.data,
                                      self.source_path,output,'cpu',epochs=5)
        self.assertEqual(result['best_epoch'],0)
        self.assertFalse(result['improved'])
        self.assertEqual(result['epochs_run'],3)
        self.assertTrue(any(observed_train_updates))
        for name,value in original.items():
            self.assertTrue(torch.equal(value,self.model.state_dict()[name]),name)
        checkpoint=torch.load(output/'J2_best.pt',map_location='cpu',weights_only=True)
        self.assertEqual(checkpoint['best_epoch'],0)
        self.assertEqual(checkpoint['config']['seed'],2026)
        self.assertEqual(checkpoint['config']['source_sha256'],j2.common.file_hash(self.source_path))
        self.assertTrue(torch.equal(checkpoint['feature_mean'],self.source['feature_mean']))
        self.assertTrue(torch.equal(checkpoint['feature_std'],self.source['feature_std']))
        self.assertEqual(checkpoint['source_feature_metadata'],self.source['feature_metadata'])
        self.assertEqual(checkpoint['class_to_idx'],self.data['class_to_idx'])
        self.assertTrue(result['no_test_inference'])
        with self.assertRaises(FileExistsError):
            j2.train_finetune(self.model,self.loader,self.loader,self.source,self.data,
                              self.source_path,output,'cpu',epochs=1)
        with self.assertRaises(AssertionError):
            j2.train_finetune(self.model,self.loader,self.loader,self.source,self.data,
                              self.source_path,self.root/'too-many','cpu',epochs=11)


if __name__=='__main__': unittest.main()
