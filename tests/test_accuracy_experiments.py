"""J0 contracts on synthetic images and a tiny encoder; no pretrained downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import inspect
import ast

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import nbformat

import accuracy_experiments as j0


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pretrained_cfg = {'source': 'synthetic-test-only'}
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        # Crops/flips change spatial samples so independently augmented views differ.
        return images[:, :, ::14, ::14].flatten(1)[:, :384] + self.anchor


class AccuracyExperimentsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        for name in ('cache', 'checkpoints', 'metrics', 'plots', 'submissions'):
            (self.output / name).mkdir(parents=True)
        self.data_root = self.root / 'data'
        rng = np.random.default_rng(2026)
        for split in ('train', 'val'):
            for label in range(20):
                folder = self.data_root / split / f'{label:03d}_car'
                folder.mkdir(parents=True)
                Image.fromarray(rng.integers(0, 256, (75, 83, 3), dtype=np.uint8)).save(folder / 'sample.png')
        (self.data_root / 'test').mkdir()
        for name in ('0001', '0002'):
            Image.fromarray(rng.integers(0, 256, (75, 83, 3), dtype=np.uint8)).save(self.data_root / 'test' / f'{name}.png')
        (self.data_root / 'sample_submission.csv').write_text('image_id,label\n0002,0\n0001,0\n')
        self.data = j0.common.load_data(self.data_root, (20, 20, 2))
        self.model = TinyEncoder().eval()

    def tearDown(self):
        self.temp.cleanup()

    def extract(self, **kwargs):
        return j0.get_features(self.model, self.data, 224, self.output, 'cpu', batch_size=8, **kwargs)

    def test_view_alignment_reproducibility_and_weight_cache_identity(self):
        payload, key = self.extract()
        self.assertEqual(payload['X_train'].shape, (100, 384))
        self.assertEqual(payload['metadata']['torchvision'], str(j0.torchvision.__version__))
        self.assertEqual(payload['metadata']['pillow'], Image.__version__)
        self.assertEqual(payload['X_val'].shape, (20, 384))
        self.assertTrue(torch.equal(payload['y_train'], self.data['labels'][0].repeat(5)))
        self.assertTrue(torch.equal(payload['y_val'], self.data['labels'][1]))
        deterministic, _ = j0.make_transforms(224)
        with Image.open(self.data['paths'][0][0]) as image:
            with torch.inference_mode():
                expected = self.model(deterministic(image.convert('RGB')).unsqueeze(0))[0]
        self.assertTrue(torch.equal(payload['X_train'][0], expected))
        self.assertFalse(torch.equal(payload['X_train'][20:40], payload['X_train'][40:60]))
        calls = self.model.calls
        # A valid cache is sufficient; encoder must not execute on a hit.
        with patch.object(self.model, 'forward', side_effect=AssertionError('unexpected cache miss')):
            cached, hit_key = self.extract()
        self.assertEqual(hit_key, key)
        self.assertEqual(calls, self.model.calls)
        self.assertTrue(torch.equal(cached['X_train'], payload['X_train']))
        regenerated, same_key = self.extract(force=True)
        self.assertEqual(same_key, key)
        self.assertTrue(torch.equal(regenerated['X_train'], payload['X_train']))
        with torch.no_grad():
            self.model.anchor.add_(1)
        changed, changed_key = self.extract()
        self.assertNotEqual(key, changed_key)
        self.assertNotEqual(payload['metadata']['weight_sha256'], changed['metadata']['weight_sha256'])
        self.assertTrue(torch.allclose(changed['X_train'], payload['X_train'] + 1))

    def test_corrupt_cache_regenerates_and_resolution_is_distinct(self):
        payload, key = self.extract()
        path = self.output / 'cache' / f'J0_S14_224_{key}.pt'
        payload['y_train'] = torch.zeros(100, dtype=torch.long)
        torch.save(payload, path)
        calls = self.model.calls
        regenerated, same_key = self.extract()
        self.assertEqual(same_key, key)
        self.assertGreater(self.model.calls, calls)
        self.assertTrue(torch.equal(regenerated['y_train'], self.data['labels'][0].repeat(5)))
        high, high_key = j0.get_features(self.model, self.data, 336, self.output, 'cpu', batch_size=8)
        self.assertNotEqual(key, high_key)
        self.assertEqual(high['metadata']['size'], 336)
        self.assertEqual(high['X_train'].shape, (100, 384))

    def test_train_only_statistics_seed_checkpoint_and_submission_order(self):
        features, key = self.extract()
        # Extreme validation features must not change the fitted training statistics.
        expected_mean = features['X_train'].mean(0)
        expected_std = features['X_train'].std(0, unbiased=False).clamp_min(1e-6)
        features['X_val'] = features['X_val'] + 1000
        first = j0.train_head(features, 42, 224, key, self.data, self.output, 'cpu')
        saved = torch.load(first['checkpoint'], map_location='cpu', weights_only=True)
        self.assertTrue(torch.equal(saved['feature_mean'], expected_mean))
        self.assertTrue(torch.equal(saved['feature_std'], expected_std))
        self.assertEqual(saved['record']['head_seed'], 42)
        self.assertEqual(saved['record']['augmentation_seed'], 123)
        self.assertEqual(saved['record']['cache_id'], key)
        self.assertEqual(saved['head_state_dict']['0.weight'].shape, (128, 384))
        self.assertEqual(saved['class_to_idx'], self.data['class_to_idx'])
        head = j0.build_head()
        head.load_state_dict(saved['head_state_dict'], strict=True)
        history = pd.read_csv(self.output / 'metrics' / 'J0_S14_224_aug123_head42_history.csv')
        self.assertAlmostEqual(history.val_accuracy.max(), first['accuracy'])
        self.assertEqual(int(history.loc[history.val_accuracy.idxmax(), 'epoch']), first['best_epoch'])
        second = j0.train_head(features, 42, 224, key, self.data, self.output, 'cpu')
        repeated = torch.load(second['checkpoint'], map_location='cpu', weights_only=True)
        for name in saved['head_state_dict']:
            self.assertTrue(torch.equal(saved['head_state_dict'][name], repeated['head_state_dict'][name]))
        other = j0.train_head(features, 2026, 224, key, self.data, self.output, 'cpu')
        other_saved = torch.load(other['checkpoint'], map_location='cpu', weights_only=True)
        self.assertFalse(torch.equal(saved['head_state_dict']['0.weight'], other_saved['head_state_dict']['0.weight']))
        self.assertTrue(torch.equal(saved['feature_mean'], other_saved['feature_mean']))
        submission_path = j0.generate_submission(self.model, other, self.data, self.output, 'cpu', 2)
        submission = pd.read_csv(submission_path, dtype={'image_id': str})
        self.assertEqual(submission.image_id.tolist(), ['0002', '0001'])
        self.assertEqual(list(submission.columns), ['image_id', 'label'])
        self.assertTrue(submission.label.between(0, 19).all())
        self.assertTrue(submission.image_id.is_unique)
        with self.assertRaises(FileExistsError):
            j0.generate_submission(self.model, other, self.data, self.output, 'cpu', 2)
        with torch.no_grad():
            self.model.anchor.add_(1)
        with self.assertRaises(AssertionError):
            j0.generate_submission(self.model, other, self.data, self.output, 'cpu', 2)

    def test_shared_cache_across_run_directories(self):
        shared = self.root / 'shared-cache'
        payload, key = j0.get_features(self.model, self.data, 224, self.root / 'run-a',
                                       'cpu', batch_size=8, cache_root=shared)
        self.assertTrue((shared / f'J0_S14_224_{key}.pt').is_file())
        with patch.object(self.model, 'forward', side_effect=AssertionError('unexpected extraction')):
            reused, reused_key = j0.get_features(self.model, self.data, 224, self.root / 'run-b',
                                                  'cpu', batch_size=8, cache_root=shared)
        self.assertEqual(key, reused_key)
        self.assertTrue(torch.equal(payload['X_train'], reused['X_train']))
        self.assertFalse((self.root / 'run-b' / 'cache').exists())

    def test_training_config_controls_actual_training(self):
        features, key = self.extract()
        settings = {'epochs':2, 'patience':1, 'batch_size':32, 'lr':0.0002, 'weight_decay':0.02}
        with patch.dict(j0.TRAIN_CONFIG, settings), \
                patch.object(j0, 'DataLoader', wraps=j0.DataLoader) as loader, \
                patch.object(torch.optim, 'AdamW', wraps=torch.optim.AdamW) as optimizer:
            result = j0.train_head(features, 42, 224, key, self.data, self.output, 'cpu')
        self.assertEqual(loader.call_args.kwargs['batch_size'],32)
        self.assertEqual(optimizer.call_args.kwargs['lr'],0.0002)
        self.assertEqual(optimizer.call_args.kwargs['weight_decay'],0.02)
        self.assertLessEqual(result['epochs_run'],2)
        self.assertEqual(result['training_config'],settings)
        self.assertEqual(j0.TRAIN_CONFIG['epochs'],35)

    def test_self_contained_notebook_sequential_utilities(self):
        notebook_path = Path(j0.__file__).with_name('J0_Resolution_Improvement.ipynb')
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        implementation = None
        for cell in notebook.cells:
            if cell.cell_type != 'code' or cell.source.startswith('%'):
                continue
            compile(cell.source, 'J0-notebook', 'exec')
            if 'def get_features(' in cell.source:
                implementation = cell.source
        self.assertIsNotNone(implementation)
        self.assertNotIn('import preprocessing_benchmark', implementation)
        namespace = {'__name__':'j0_notebook_synthetic_test'}
        # Preserve imports from the earlier configuration cell without executing
        # user paths, optional Drive mounting or any experiment main invocation.
        earlier_imports = []
        for cell in notebook.cells:
            if cell.cell_type != 'code' or cell.source.startswith('%'):
                continue
            if cell.source == implementation:
                break
            earlier_imports.extend(node for node in ast.parse(cell.source).body
                                   if isinstance(node,(ast.Import,ast.ImportFrom)))
        exec(compile(ast.Module(body=earlier_imports,type_ignores=[]), 'J0-config-imports', 'exec'), namespace)
        # Installation, user paths and real main/model invocation are intentionally skipped.
        exec(compile(implementation, 'J0-notebook', 'exec'), namespace)
        self.assertEqual(inspect.signature(namespace['main']).parameters['argv'].default, None)
        data = namespace['common'].load_data(self.data_root, (20,20,2))
        self.assertEqual(data['ids'], ['0002','0001'])
        namespace['TRAIN_CONFIG'] = dict(namespace['TRAIN_CONFIG'], epochs=2, patience=1)
        records = []
        for size in (224,336):
            features, key = namespace['get_features'](self.model,data,size,self.output,'cpu',batch_size=8)
            self.assertEqual(features['X_train'].shape, (100,384))
            self.assertTrue(torch.equal(features['y_train'],data['labels'][0].repeat(5)))
            for seed in namespace['SEEDS']:
                records.append(namespace['train_head'](features,seed,size,key,data,self.output,'cpu'))
        frame, summary = namespace['compare'](records,self.output)
        self.assertEqual(len(frame),6)
        self.assertEqual(summary.n.tolist(),[3,3])
        for size in (224,336):
            selected = frame[frame.resolution==size]
            self.assertAlmostEqual(summary.loc[size,'accuracy_mean'],selected.accuracy.mean())
            self.assertAlmostEqual(summary.loc[size,'accuracy_std'],selected.accuracy.std(ddof=1))
            self.assertAlmostEqual(summary.loc[size,'macro_f1_std'],selected.macro_f1.std(ddof=1))
        resolution = int(summary.sort_values(['accuracy_mean','macro_f1_mean'],
                                              ascending=False,kind='stable').index[0])
        chosen = next(record for record in records if record['resolution']==resolution and record['head_seed']==2026)
        path = namespace['generate_submission'](self.model,chosen,data,self.output,'cpu',2)
        submission = pd.read_csv(path,dtype={'image_id':str})
        self.assertEqual(submission.image_id.tolist(),data['ids'])
        self.assertTrue(submission.label.between(0,19).all())
        self.assertTrue((self.output/'plots'/'J0_seed_comparison.png').is_file())


if __name__ == '__main__':
    unittest.main()
