"""Synthetic sequential notebook test; never downloads or claims real I0 results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image
import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / 'I0_DINOv2_ViTB14.ipynb'

class Encoder(nn.Module):
    embed_dim = 768
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x[:, :, ::14, ::14].flatten(1)[:, :768] + self.anchor

class I0NotebookTest(unittest.TestCase):
    def test_sequential_execution_and_artifact_contracts(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for c in notebook.cells:
            if c.cell_type == 'code' and not c.source.startswith('%'):
                compile(c.source, 'I0-notebook', 'exec')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            rng = np.random.default_rng(100)
            # Keep the real split counts, transforms and class prefix assertions.
            for split, count in [('train',35),('val',10)]:
                for label in range(20):
                    folder = data / split / f'{label:03d}_car'
                    folder.mkdir(parents=True)
                    for i in range(count):
                        Image.fromarray(rng.integers(0,256,(24,28,3),dtype=np.uint8)).save(folder / f'{i:03d}.png')
            (data/'test').mkdir()
            for i in range(700):
                Image.fromarray(rng.integers(0,256,(24,28,3),dtype=np.uint8)).save(data/'test'/f'{i:03d}.png')
            import pandas as pd
            ids = [f'{i:03d}' for i in reversed(range(700))]
            pd.DataFrame({'image_id':ids,'label':0}).to_csv(data/'sample_submission.csv',index=False)
            scope = {}
            with patch('torch.hub.load', return_value=Encoder()) as hub:
                for c in notebook.cells:
                    if c.cell_type != 'code' or c.source.startswith('%'): continue
                    code = c.source
                    if 'USE_DRIVE = False' in code:
                        code = code.replace("Path('/content/car_data')", repr(data))
                        code = code.replace("Path('/content')", repr(root))
                        # Path repr needs Path in the execution scope (already imported).
                        code = code.replace('PosixPath(', 'Path(')
                        code = code.replace('EPOCHS = 35', 'EPOCHS = 1')
                    exec(compile(code, 'I0-notebook', 'exec'), scope)
                args, kwargs = hub.call_args
                self.assertEqual(args[1], 'dinov2_vitb14')
                self.assertTrue(kwargs['pretrained'])
            self.assertEqual(scope['HEAD_SEEDS'], [2026])
            self.assertEqual(tuple(scope['X_train'].shape), (3500,768))
            self.assertTrue(torch.equal(scope['y_train'],torch.tensor(scope['train_ds'].targets).repeat(5)))
            self.assertFalse(torch.equal(scope['X_train'][700:1400],scope['X_train'][1400:2100]))
            self.assertTrue(torch.equal(scope['feature_mean'],scope['X_train'].mean(0)))
            self.assertFalse(scope['backbone'].training)
            self.assertFalse(any(p.requires_grad for p in scope['backbone'].parameters()))
            record = scope['records'][0]
            saved = torch.load(record['checkpoint'],weights_only=True)
            self.assertEqual(saved['head_state_dict']['0.weight'].shape,(128,768))
            self.assertEqual(saved['seed_head'],2026)
            head=scope['build_head']()
            head.load_state_dict(saved['head_state_dict'],strict=True)
            submission=pd.read_csv(scope['submission_paths'][0],dtype={'image_id':str})
            self.assertEqual(submission.image_id.tolist(),ids)
            self.assertEqual(list(submission.columns),['image_id','label'])
            self.assertTrue(submission.label.between(0,19).all())
            with self.assertRaises(FileExistsError): scope['generate_submission'](record)
            self.assertFalse((scope['OUTPUT_DIR']/'I0_multiseed_summary.csv').exists())
            # Explicitly exercise two additional fresh heads on the same embeddings.
            mean_before = scope['feature_mean'].clone()
            for seed in (123,42):
                scope['records'].append(scope['train_classifier'](seed))
            self.assertTrue(torch.equal(mean_before,scope['feature_mean']))
            evaluation = next(c.source for c in notebook.cells
                              if c.cell_type=='code' and c.source.startswith('results = pd.DataFrame(records)'))
            exec(evaluation,scope)
            summary = pd.read_csv(scope['OUTPUT_DIR']/'I0_multiseed_summary.csv',index_col=0)
            self.assertAlmostEqual(summary.loc['std','accuracy'],scope['results'].accuracy.std(ddof=1))
            # Changing only a classifier seed must preserve cache identity.
            signature = scope['FEATURES_SIGNATURE']
            scope['HEAD_SEEDS']=[123,42,2026]
            with patch.dict(scope, extract=lambda *a: (_ for _ in ()).throw(AssertionError('cache miss'))):
                cached,key,_=scope['get_features']()
            self.assertEqual(key,signature)
            meta=scope['feature_metadata']()
            self.assertTrue(scope['valid_cache'](cached,meta))
            stale=dict(cached); stale['metadata'] = dict(meta, backbone='dinov2_vits14')
            self.assertFalse(scope['valid_cache'](stale,meta))
            stale=dict(cached); stale['X_train']=torch.zeros(3500,384)
            self.assertFalse(scope['valid_cache'](stale,meta))
            stale=dict(cached); stale['y_train']=torch.zeros(3500,dtype=torch.long)
            self.assertFalse(scope['valid_cache'](stale,meta))
            # Corrupt cache regenerates instead of silently accepting bad features.
            torch.save(stale,scope['CACHE_PATH'])
            calls=[]
            def fast_extract(loader, desc):
                calls.append(desc)
                return torch.zeros(len(loader.dataset),768)
            with patch.dict(scope,extract=fast_extract):
                regenerated,key,_=scope['get_features']()
            self.assertEqual(len(calls),7)
            self.assertTrue(scope['valid_cache'](regenerated,meta))
            scope['SEED_AUG']=42
            self.assertNotEqual(scope['metadata_digest'](scope['feature_metadata']()),signature)

if __name__ == '__main__': unittest.main()
