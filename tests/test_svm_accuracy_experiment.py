"""J1 train-only CV/grouping tests with synthetic features; no real validation runs."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import svm_accuracy_experiment as j1


class SvmAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(2026)
        self.n = 60
        self.labels = np.repeat(np.arange(20),3)
        self.x = self.rng.normal(size=(self.n*5,384)).astype(np.float32)
        self.x[:,0] = np.arange(self.n*5)
        self.data = {'paths':[[Path(f'train/{i}.jpg') for i in range(self.n)],
                              [Path(f'val/{i}.jpg') for i in range(20)],
                              [Path('test/a.jpg'),Path('test/b.jpg')]],
                     'labels':[torch.tensor(self.labels),torch.arange(20)],
                     'class_to_idx':{f'{i:03d}_car':i for i in range(20)},
                     'manifest':[{'split':'synthetic','files':[]}], 'ids':['b','a']}
        self.metadata = {'model':j1.MODEL_NAME,'size':224,'layout':'view-major',
                         'original_views':1,'augmented_views':4,'aug_seed':123,
                         'class_to_idx':self.data['class_to_idx'],'manifest':self.data['manifest']}
        self.payload = {'metadata':self.metadata,'X_train':torch.from_numpy(self.x.copy()),
                        'X_val':torch.full((20,384),10000,dtype=torch.float32),
                        'y_train':torch.tensor(np.tile(self.labels,5)), 'y_val':torch.arange(20)}
        self.key = j1.common.digest(self.metadata)

    def tearDown(self):
        self.temp.cleanup()

    def test_cv_excludes_all_siblings_and_scores_originals_only(self):
        original_labels = self.labels
        fitted = []
        scored = []
        n = self.n
        class TrackedClassifier:
            def fit(self, x, y):
                self.indices = x[:,0].astype(int)
                self.originals = set(self.indices % n)
                np.testing.assert_array_equal(y,original_labels[self.indices % n])
                for index in self.originals:
                    assert set(self.indices[self.indices % n == index]) == {index + view*n for view in range(5)}
                fitted.append(self.indices)
                return self
            def predict(self, x):
                indices = x[:,0].astype(int)
                assert (indices < n).all()
                assert not self.originals.intersection(indices)
                scored.append(indices)
                return original_labels[indices]
        with patch.object(j1,'make_classifier',side_effect=lambda c:TrackedClassifier()):
            selected, diagnostics, summary, assignment = j1.select_c_train_only(self.x,self.labels)
        self.assertEqual(selected,0.1)  # Smaller C resolves a complete CV tie.
        self.assertEqual(len(fitted),9)
        self.assertEqual(len(scored),9)
        self.assertEqual(len(diagnostics),9)
        self.assertEqual(sorted(np.unique(assignment).tolist()),[0,1,2])
        self.assertTrue((diagnostics.sibling_overlap==0).all())
        for fold in range(3):
            heldout = np.flatnonzero(assignment==fold)
            for predicted_indices in scored[fold*3:(fold+1)*3]:
                np.testing.assert_array_equal(predicted_indices,heldout)
        self.assertEqual(summary.index.tolist(),[0.1,1.0,10.0])
        with self.assertRaises(AssertionError):
            j1.select_c_train_only(self.x,self.labels,c_values=(1,10,100))

    def test_scalers_fit_each_fold_and_selected_model_uses_training_only(self):
        classifiers = []
        original_factory = j1.make_classifier
        def factory(c):
            estimator = original_factory(c)
            classifiers.append(estimator)
            return estimator
        with patch.object(j1,'make_classifier',side_effect=factory):
            selected, diagnostics, summary, assignment = j1.select_c_train_only(self.x,self.labels)
        for fold in range(3):
            fit_original = np.flatnonzero(assignment!=fold)
            expected = self.x[j1.view_rows(fit_original,self.n)].mean(0,dtype=np.float64)
            for classifier in classifiers[fold*3:(fold+1)*3]:
                np.testing.assert_allclose(classifier['scale'].mean_,expected,rtol=0,atol=1e-10)
        output = self.root/'result'
        result = j1.fit_selected_and_evaluate(self.payload,self.data,224,self.key,output)
        final = joblib.load(output/'J1_svm.joblib')  # This test's own trusted local artifact.
        np.testing.assert_allclose(final['scale'].mean_,self.x.mean(0,dtype=np.float64),rtol=0,atol=1e-10)
        self.assertEqual(final['svm'].C,selected)
        self.assertFalse(result['test_predictions_generated'])
        self.assertTrue(result['no_kaggle_submission'])
        predictions = pd.read_csv(output/'J1_validation_predictions.csv')
        self.assertEqual(predictions.path.tolist(),[str(p) for p in self.data['paths'][1]])
        self.assertTrue(predictions.predicted_label.between(0,19).all())
        self.assertEqual(len(predictions),20)
        selection = json.loads((output/'J1_selection.json').read_text())
        self.assertEqual(selection['C_candidates'],[0.1,1.0,10.0])
        self.assertEqual(selection['source_cache_id'],self.key)
        with self.assertRaises(FileExistsError):
            j1.fit_selected_and_evaluate(self.payload,self.data,224,self.key,output)
        # Changing provided-validation labels cannot influence C selection.
        altered = dict(self.payload,y_val=torch.tensor(np.roll(np.arange(20),1)))
        other = j1.fit_selected_and_evaluate(altered,self.data,224,self.key,self.root/'other')
        self.assertEqual(result['C'],other['C'])

    def test_completed_cache_metadata_and_fold_alignment_guards(self):
        j1.validate_features(self.payload,self.data,224,self.key)
        stale = dict(self.payload,y_train=torch.zeros(self.n*5,dtype=torch.long))
        with self.assertRaises(AssertionError): j1.validate_features(stale,self.data,224,self.key)
        with self.assertRaises(AssertionError): j1.validate_features(self.payload,self.data,336,self.key)
        with self.assertRaises(AssertionError): j1.validate_features(self.payload,self.data,224,'bad-key')
        run = self.root/'J0'/'run'
        (run/'metrics').mkdir(parents=True)
        with self.assertRaises(FileNotFoundError): j1.load_completed_features(run,self.root)
        (run/'summary.json').write_text(json.dumps({'selection':{'resolution':224}}))
        record = {'cache_id':self.key,'resolution':224}
        (run/'metrics'/'J0_S14_224_aug123_head2026.json').write_text(json.dumps(record))
        shared = run.parent/'cache'
        shared.mkdir()
        torch.save(self.payload,shared/f'J0_S14_224_{self.key}.pt')
        (self.root/'sample_submission.csv').write_text('image_id,label\nb,0\na,0\n')
        with patch.object(j1.common,'load_data',return_value=self.data):
            loaded,data,source,path = j1.load_completed_features(run,self.root)
        self.assertEqual(source,record)
        self.assertEqual(path.parent,shared)
        self.assertTrue(torch.equal(loaded['X_train'],self.payload['X_train']))
        with self.assertRaises(AssertionError): j1.load_completed_features(run,self.root,resolution=336)
        with self.assertRaises(AssertionError): j1.view_rows([0,0],self.n)


if __name__=='__main__': unittest.main()
