"""J1: one train-only-selected RBF-SVM candidate on existing frozen J0 features.

No backbone inference, test prediction, external data or Kaggle API calls. Cached
features are raw; every CV fold fits its own scaler on training images' views.
"""
from pathlib import Path
import argparse
import json
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import preprocessing_benchmark as common

C_VALUES = (0.1, 1.0, 10.0)
CV_SEED = 123
N_SPLITS = 3
N_VIEWS = 5
FEATURE_DIM = 384
MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'


def view_rows(original_indices, n_original, n_views=N_VIEWS):
    """Indices for view-major blocks; each image's siblings stay in one fold."""
    indices = np.asarray(original_indices, dtype=np.int64)
    assert indices.ndim == 1 and len(np.unique(indices)) == len(indices)
    assert np.all((indices >= 0) & (indices < n_original))
    return np.concatenate([indices + view * n_original for view in range(n_views)])


def make_classifier(c):
    return Pipeline([('scale', StandardScaler()),
                     ('svm', SVC(C=float(c), kernel='rbf', gamma='scale'))])


def validate_features(payload, data, resolution, cache_id):
    metadata = payload['metadata']
    assert common.digest(metadata) == cache_id, 'Cache signature mismatch'
    assert metadata['model'] == MODEL_NAME and metadata['size'] == resolution
    assert metadata['layout'] == 'view-major'
    assert metadata['original_views'] == 1 and metadata['augmented_views'] == 4
    assert metadata['aug_seed'] == 123
    assert metadata['class_to_idx'] == data['class_to_idx']
    assert metadata['manifest'] == data['manifest'], 'Dataset content/order differs from J0 cache'
    n_original, n_val = map(len, data['paths'][:2])
    for key, n in [('X_train', n_original * N_VIEWS), ('X_val', n_val)]:
        x = payload[key]
        assert torch.is_tensor(x) and x.dtype == torch.float32
        assert x.shape == (n, FEATURE_DIM) and torch.isfinite(x).all()
    assert torch.equal(payload['y_train'], data['labels'][0].repeat(N_VIEWS)), 'View-major labels misaligned'
    assert torch.equal(payload['y_val'], data['labels'][1])


def load_completed_features(j0_run, data_root, resolution=None):
    j0_run = Path(j0_run)
    summary_path = j0_run / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'J0 must finish first: missing {summary_path}')
    summary = json.loads(summary_path.read_text())
    selected_resolution = int(summary['selection']['resolution'])
    if resolution is None:
        resolution = selected_resolution
    assert resolution == selected_resolution, 'J1 uses J0-selected resolution; no additional resolution search'
    assert resolution in (224, 336)
    record_path = j0_run / 'metrics' / f'J0_S14_{resolution}_aug123_head2026.json'
    record = json.loads(record_path.read_text())
    cache_id = record['cache_id']
    filename = f'J0_S14_{resolution}_{cache_id}.pt'
    # Initial J0 runs used run/cache; later runs use a shared parent/cache.
    candidates = [directory / filename for directory in (j0_run / 'cache', j0_run.parent / 'cache')]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(f'Missing J0 feature cache {filename}; preserve its original cache directory')
    payload = torch.load(matches[0], map_location='cpu', weights_only=True)
    data = common.load_data(data_root)
    assert (Path(data_root) / 'sample_submission.csv').is_file()
    validate_features(payload, data, resolution, cache_id)
    return payload, data, record, matches[0]


def select_c_train_only(x_train, y_original, c_values=C_VALUES):
    """Select C entirely inside the original training split.

    A fold contains original image indices. Its fitting rows include all five
    training views. Its scoring rows are held-out originals from block zero,
    never siblings of fitting images. No provided-validation features/labels enter.
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    y_original = np.asarray(y_original, dtype=np.int64)
    n_original = len(y_original)
    assert x_train.ndim == 2 and len(x_train) == n_original * N_VIEWS
    assert np.isfinite(x_train).all()
    assert tuple(float(c) for c in c_values) == C_VALUES, 'Only the three predeclared C values are allowed'
    assert min(np.unique(y_original, return_counts=True)[1]) >= N_SPLITS
    expanded_labels = np.tile(y_original, N_VIEWS)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    folds = list(splitter.split(np.zeros(n_original), y_original))
    assignment = np.full(n_original, -1, dtype=np.int64)
    rows = []
    for fold, (fit_original, score_original) in enumerate(folds):
        assignment[score_original] = fold
        fit_rows = view_rows(fit_original, n_original)
        assert not np.intersect1d(fit_rows % n_original, score_original).size
        assert np.all(score_original < n_original), 'CV scores only original-view block zero'
        for c in c_values:
            classifier = make_classifier(c)
            classifier.fit(x_train[fit_rows], expanded_labels[fit_rows])
            pred = classifier.predict(x_train[score_original])
            rows.append({'C':float(c), 'fold':fold, 'fit_original_images':len(fit_original),
                         'fit_view_rows':len(fit_rows), 'scored_original_images':len(score_original),
                         'sibling_overlap':0, 'accuracy':float(accuracy_score(y_original[score_original], pred)),
                         'macro_f1':float(f1_score(y_original[score_original], pred,
                            labels=np.unique(y_original), average='macro', zero_division=0))})
    assert (assignment >= 0).all()
    diagnostics = pd.DataFrame(rows)
    aggregate = diagnostics.groupby('C').agg(accuracy_mean=('accuracy','mean'),
        accuracy_std=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
    # Predeclared ranking; smaller C resolves a complete tie.
    ranking = aggregate.reset_index().sort_values(['accuracy_mean','macro_f1_mean','C'],
                                                  ascending=[False,False,True],kind='stable')
    selected = float(ranking.iloc[0]['C'])
    return selected, diagnostics, aggregate, assignment


def fit_selected_and_evaluate(payload, data, resolution, cache_id, output):
    """One provided-validation evaluation after C selection; no test evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    x_train = payload['X_train'].numpy()
    y_original = np.asarray(data['labels'][0], dtype=np.int64)
    c, folds, cv_summary, assignment = select_c_train_only(x_train, y_original)
    folds.to_csv(output / 'J1_cv_folds.csv', index=False)
    cv_summary.to_csv(output / 'J1_cv_summary.csv')
    pd.DataFrame({'path':[str(p) for p in data['paths'][0]], 'label':y_original,
                  'heldout_fold':assignment}).to_csv(output / 'J1_original_image_folds.csv', index=False)
    selection = {'experiment':'J1', 'model':'RBF SVM', 'C':c, 'gamma':'scale',
                 'C_candidates':list(C_VALUES), 'cv_seed':CV_SEED, 'cv_folds':N_SPLITS,
                 'resolution':resolution, 'source_cache_id':cache_id,
                 'scaler':'StandardScaler fitted within each fold; final scaler uses all training views',
                 'CV_scoring':'held-out deterministic original views only; all sibling views excluded from fit',
                 'selection_rule':'mean CV accuracy, mean CV macro-F1, smaller C',
                 'sklearn':sklearn.__version__, 'feature_metadata':payload['metadata']}
    # Persist the training-only decision before accessing held-out validation arrays.
    (output / 'J1_selection.json').write_text(json.dumps(selection, indent=2))
    classifier = make_classifier(c)
    classifier.fit(x_train, np.tile(y_original,N_VIEWS))
    joblib.dump(classifier, output / 'J1_svm.joblib', compress=1)
    validation_pred = classifier.predict(payload['X_val'].numpy()).astype(np.int64)
    y_val = np.asarray(payload['y_val'], dtype=np.int64)
    assert len(validation_pred) == len(data['paths'][1]) and np.all((validation_pred>=0)&(validation_pred<20))
    measured = {'accuracy':float(accuracy_score(y_val, validation_pred)),
                'macro_f1':float(f1_score(y_val, validation_pred,labels=list(range(20)),average='macro',zero_division=0)),
                'correct':int((validation_pred==y_val).sum()),'validation_count':len(y_val),
                'C':c,'resolution':resolution,'test_predictions_generated':False,
                'external_training_data':False,'no_kaggle_submission':True}
    (output / 'J1_metrics.json').write_text(json.dumps(measured, indent=2))
    pd.DataFrame({'path':[str(p) for p in data['paths'][1]],'true_label':y_val,
                  'predicted_label':validation_pred}).to_csv(output / 'J1_validation_predictions.csv', index=False)
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--j0-run',required=True,help='Completed J0 UUID run directory')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--resolution',type=int,choices=(224,336),default=None,
                        help='Must match J0 selected resolution; by default read from summary.json')
    parser.add_argument('--output-root',default='outputs/J1')
    args = parser.parse_args(argv)
    features,data,record,cache_path = load_completed_features(args.j0_run,args.data_root,args.resolution)
    output = Path(args.output_root) / uuid.uuid4().hex[:12]
    result = fit_selected_and_evaluate(features,data,record['resolution'],record['cache_id'],output)
    print(json.dumps({'output':str(output.resolve()),'source_cache':str(cache_path),'metrics':result},indent=2))


if __name__ == '__main__':
    main()
