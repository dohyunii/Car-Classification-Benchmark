import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
import preprocessing_benchmark as b

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1),requires_grad=False)
        self.calls = 0
    def forward(self,x):
        self.calls += 1
        # Spatial signal makes crop differences observable.
        return x[:,:,::16,::16].flatten(1)[:,:384] + self.anchor

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/'data'
        rng = np.random.default_rng(9)
        for split in ('train','val'):
            for i in range(20):
                path=self.data/split/f'{i:03d}_car'/'image.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.integers(0,256,(270,300,3),dtype=np.uint8)).save(path)
        (self.data/'test').mkdir()
        for name in ('001','002'):
            Image.fromarray(rng.integers(0,256,(270,300,3),dtype=np.uint8)).save(self.data/'test'/f'{name}.png')
        (self.data/'sample_submission.csv').write_text('image_id,label\n002,0\n001,0\n')
        self.loaded=b.load_data(self.data,(20,20,2))
        self.model=TinyEncoder()
        self.output=self.root/'outputs'
    def tearDown(self):
        self.tmp.cleanup()
    def cfg(self,id='P0',**kw):
        return b.Config(experiment_id=id,expected_counts=(20,20,2),epochs=2,**kw)
    def features(self,cfg,force=False):
        return b.get_features(cfg,self.loaded,self.model,self.output,'cpu',force)
    def test_views_reproducibility_cache_identity_and_recovery(self):
        for name,views in [('P0',1),('P1',1),('P2',1),('P3',5),('P4',5)]:
            cfg=self.cfg(name)
            features,key=self.features(cfg)
            self.assertEqual(tuple(features['X_train'].shape),(20*views,384))
            self.assertTrue(torch.equal(features['y_train'],self.loaded['labels'][0].repeat(views)))
            calls=self.model.calls
            cached,key2=self.features(self.cfg(name,head_seed=2026))
            self.assertEqual(key,key2); self.assertEqual(calls,self.model.calls)
            fresh,_=self.features(cfg,True)
            self.assertTrue(torch.equal(features['X_train'],fresh['X_train']))
            if views==5:
                self.assertFalse(torch.equal(features['X_train'][20:40],features['X_train'][40:60]))
            self.assertTrue(torch.equal(features['X_val'],fresh['X_val']))
        cfg=self.cfg('P3')
        _,key=self.features(cfg)
        _,different=self.features(self.cfg('P3',augmentation_seed=42))
        self.assertNotEqual(key,different)
        path=self.output/'cache'/f'P3_{key}.pt'
        corrupt=torch.load(path,weights_only=True)
        corrupt['y_train']=torch.zeros(100,dtype=torch.long)
        torch.save(corrupt,path)
        calls=self.model.calls
        recovered,_=self.features(cfg)
        self.assertGreater(self.model.calls,calls)
        self.assertTrue(torch.equal(recovered['y_train'],self.loaded['labels'][0].repeat(5)))
        old=b.feature_metadata(cfg,self.loaded,self.model)
        with torch.no_grad(): self.model.anchor.add_(1)
        self.assertNotEqual(b.digest(old),b.digest(b.feature_metadata(cfg,self.loaded,self.model)))
        original=b.file_hash(self.loaded['paths'][0][0])
        Image.new('RGB',(270,300),(1,2,3)).save(self.loaded['paths'][0][0])
        changed=b.load_data(self.data,(20,20,2))
        self.assertNotEqual(original,b.file_hash(changed['paths'][0][0]))
        self.assertNotEqual(self.loaded['manifest'],changed['manifest'])
    def test_standardization_and_fresh_head(self):
        features,_=self.features(self.cfg())
        standardized,mean,std=b.standardize(features)
        self.assertTrue(torch.allclose(mean,features['X_train'].mean(0)))
        features['X_val'] = torch.full_like(features['X_val'], 10000)
        _,new_mean,new_std=b.standardize(features)
        self.assertTrue(torch.equal(mean,new_mean)); self.assertTrue(torch.equal(std,new_std))
        first=b.train_head(self.cfg(),features,'cpu')
        second=b.train_head(self.cfg(),features,'cpu')
        for key in first['head_state_dict']:
            self.assertTrue(torch.equal(first['head_state_dict'][key],second['head_state_dict'][key]))
    def test_artifacts_and_summary(self):
        records=[]
        for seed in (42,123):
            records.append(b.run_experiment(self.cfg('P4',head_seed=seed),self.data,self.output,
                make_submission=True,backbone=self.model))
        frame,summary=b.compare_results(records,self.output)
        self.assertEqual(len(frame),2)
        self.assertAlmostEqual(summary.loc['P4','accuracy_std'],frame.val_accuracy.std(ddof=1))
        import pandas as pd
        for record in records:
            run=record['run_id']
            submission=pd.read_csv(self.output/'submissions'/f'{run}.csv',dtype={'image_id':str})
            self.assertEqual(submission.image_id.tolist(),['002','001'])
            self.assertEqual(list(submission.columns),['image_id','label'])
            checkpoint=torch.load(record['checkpoint'],weights_only=True)
            self.assertIn('feature_mean',checkpoint)
            saved=json.loads((self.output/'metrics'/f'{run}.json').read_text())
            self.assertEqual(saved['train_embeddings'],100)
    def test_split_and_count_guards(self):
        with self.assertRaises(AssertionError): b.load_data(self.data,(700,200,700))
        (self.data/'test'/'001.png').write_bytes(self.loaded['paths'][0][0].read_bytes())
        with self.assertRaises(AssertionError): b.load_data(self.data,(20,20,2))
    def test_notebook_structure(self):
        import nbformat
        notebook=nbformat.read(Path(b.__file__).with_name('preprocessing_experiments.ipynb'),as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):
                compile(cell.source,'notebook','exec')

if __name__=='__main__': unittest.main()
