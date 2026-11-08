"""Synthetic sequential three-head ensemble test; no real model download."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
import nbformat
from test_f0_tta_notebook import Encoder

class EnsembleNotebookTest(unittest.TestCase):
    def test_sequential_shared_features_fresh_heads_and_fixed_average(self):
        notebook=nbformat.read(Path(__file__).resolve().parents[1]/'F0_ThreeSeed_Ensemble.ipynb',as_version=4)
        nbformat.validate(notebook);torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';rng=np.random.default_rng(456)
            for split,count in [('train',35),('val',10)]:
                for label in range(20):
                    p=data/split/f'{label:03d}_car';p.mkdir(parents=True)
                    for i in range(count):Image.fromarray(rng.integers(0,256,(24,28,3),dtype=np.uint8)).save(p/f'{i}.png')
            (data/'test').mkdir()
            for i in range(2):Image.fromarray(rng.integers(0,256,(24,28,3),dtype=np.uint8)).save(data/'test'/f'{i:03d}.png')
            (data/'sample_submission.csv').write_text('image_id,label\n001,0\n000,0\n')
            scope={}
            with patch('torch.hub.load',return_value=Encoder()) as hub:
                for cell in notebook.cells:
                    if cell.cell_type!='code' or cell.source.startswith('%'):continue
                    code=cell.source
                    if code.startswith("MODE = 'train_f0'"):
                        code=code.replace('DOWNLOAD_FROM_KAGGLE = True','DOWNLOAD_FROM_KAGGLE = False')
                        code=code.replace("Path('/content/car_data')",f'Path({str(data)!r})')
                        code=code.replace("STORAGE_ROOT = Path('/content')",f'STORAGE_ROOT = Path({str(root)!r})')
                        code=code.replace('35,1e-3,1e-3,7','1,1e-3,1e-3,7')
                    exec(compile(code,'ensemble','exec'),scope)
                self.assertEqual(hub.call_count,1)
            self.assertEqual(scope['HEAD_SEEDS'],[42,123,2026])
            self.assertEqual(len(set(map(id,scope['heads']))),3)
            self.assertTrue(torch.equal(scope['ensemble_test'],scope['test_probabilities'].mean(0)))
            self.assertFalse(torch.equal(scope['heads'][0][0].weight,scope['heads'][1][0].weight))
            self.assertFalse(scope['summary']['submitted'])
            frame=pd.read_csv(scope['ensemble_csv'],dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),['001','000'])
            self.assertTrue(frame.label.between(0,19).all())
            for record in scope['records']:
                saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)
                self.assertEqual(saved['seed_head'],record['seed'])
                self.assertTrue(torch.equal(saved['feature_mean'],scope['feature_mean']))
            self.assertEqual(len(list(scope['CACHE_DIR'].glob('*.pt'))),1)
            self.assertEqual(tuple(scope['X_train'].shape),(3500,384))

if __name__=='__main__':unittest.main()
