"""Readable, full training implementation of the I6 car ensemble.

Only official pretrained backbone weights are loaded. Every classifier is freshly
initialized and trained here. No prior bundle, classifier or feature cache is read.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.error import HTTPError
import gc
import hashlib
import json
import platform
import random
import subprocess
import time
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, f1_score

HUB_REF = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
HEAD_SEEDS = [42, 123, 2026]
AUGMENTATION_SEED = 123
NUM_CLASSES = 20

# Each backbone runs once per image/view. All of its heads share those features.
BACKBONES = {
    'S': {'name': 'dinov2_vits14', 'dimension': 384, 'crop': 224, 'resize': 256, 'batch_size': 16},
    'B': {'name': 'dinov2_vitb14', 'dimension': 768, 'crop': 336, 'resize': 384, 'batch_size': 8},
    'L': {'name': 'dinov2_vitl14', 'dimension': 1024, 'crop': 336, 'resize': 384, 'batch_size': 2},
}

# S preserves the original F0 training rule. B/L follow the fixed I5/I6 recipes.
TRAINING_RULES = {
    'S_mlp': {'epochs': 35, 'patience': 7, 'selection': 'best_validation_accuracy'},
    'B_mlp': {'epochs': 3, 'patience': None, 'selection': 'final_epoch'},
    'L_mlp': {'epochs': 3, 'patience': None, 'selection': 'final_epoch'},
    'B_linear': {'epochs': 20, 'patience': None, 'selection': 'final_epoch'},
    'L_linear': {'epochs': 20, 'patience': None, 'selection': 'final_epoch'},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def backbone_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def log_event(run_dir, event, **details):
    record = {'time': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'event': event, **details}
    with (Path(run_dir) / 'execution_log.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def load_course_data(data_root, expected_counts=(700, 200, 700)):
    root = Path(data_root)
    train = datasets.ImageFolder(root / 'train')
    validation = datasets.ImageFolder(root / 'val')
    mapping = train.class_to_idx
    assert mapping == validation.class_to_idx and len(mapping) == NUM_CLASSES
    assert all(name.startswith(f'{index:03d}_') for index, name in enumerate(train.classes))

    sample = pd.read_csv(root / 'sample_submission.csv', dtype={'image_id': str})
    assert {'image_id', 'label'} <= set(sample.columns)
    assert sample.image_id.notna().all() and sample.image_id.is_unique
    test_ids = sample.image_id.tolist()
    files = [path for path in (root / 'test').iterdir()
             if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    by_id = {path.stem: path for path in files}
    assert len(by_id) == len(files) and set(by_id) == set(test_ids)
    paths = {
        'train': [Path(path) for path, _ in train.samples],
        'val': [Path(path) for path, _ in validation.samples],
        'test': [by_id[identity] for identity in test_ids],
    }
    assert tuple(len(paths[key]) for key in ('train', 'val', 'test')) == expected_counts
    manifest = {split: [{'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}
                        for path in ordered] for split, ordered in paths.items()}
    hashes = {split: {row['sha256'] for row in rows} for split, rows in manifest.items()}
    assert not (hashes['train'] & hashes['val'] or hashes['train'] & hashes['test']
                or hashes['val'] & hashes['test']), 'Duplicate image content across splits'
    return {'paths': paths, 'train_labels': torch.tensor(train.targets),
            'val_labels': torch.tensor(validation.targets), 'test_ids': test_ids,
            'class_to_idx': mapping, 'manifest': manifest}


def start_run(storage_root, data, device):
    run_id = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    run_dir = Path(storage_root) / 'student_outputs_Assignment_Training' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'torchvision': str(torchvision.__version__), 'pillow': Image.__version__,
                   'device': device, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                   'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    configuration = {'hub_ref': HUB_REF, 'backbones': BACKBONES, 'training_rules': TRAINING_RULES,
                     'head_seeds': HEAD_SEEDS, 'augmentation_seed': AUGMENTATION_SEED,
                     'learning_rate': .001, 'weight_decay': .001, 'head_batch_size': 64,
                     'class_to_idx': data['class_to_idx'], 'environment': environment,
                     'loads_previous_classifier': False, 'loads_previous_feature_cache': False}
    (run_dir / 'run_configuration.json').write_text(json.dumps(configuration, indent=2))
    (run_dir / 'dataset_manifest.json').write_text(json.dumps(data['manifest'], indent=2))
    log_event(run_dir, 'run_started', run_id=run_id, environment=environment)
    print('Output folder:', run_dir)
    print('Environment:', environment)
    return run_dir


class ImagePaths(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def create_transforms(specification):
    normalise = [transforms.ToTensor(),
                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    evaluation = transforms.Compose([
        transforms.Resize(specification['resize'], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(specification['crop']), *normalise])
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(specification['crop'], scale=(.8, 1.), ratio=(.9, 1.1),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(.5), *normalise])
    return evaluation, augmentation


def load_frozen_backbone(specification, device):
    # Download official self-supervised weights, not a trained course classifier.
    try:
        model = torch.hub.load(f'facebookresearch/dinov2:{HUB_REF}', specification['name'],
                               pretrained=True, trust_repo=True, skip_validation=True)
    except HTTPError as error:
        if error.code not in (502, 503, 504):
            raise
        # A pinned git checkout avoids transient GitHub zipball gateway failures.
        repository = Path(torch.hub.get_dir()) / ('dinov2_git_' + HUB_REF[:12])
        if not repository.exists():
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                            'https://github.com/facebookresearch/dinov2.git', str(repository)], check=True)
        subprocess.run(['git', '-C', str(repository), 'checkout', '--detach', HUB_REF], check=True)
        revision = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        assert revision == HUB_REF
        model = torch.hub.load(str(repository), specification['name'], source='local', pretrained=True)
    model.requires_grad_(False).eval()
    assert model.embed_dim == specification['dimension']
    return model.to(device)


@torch.inference_mode()
def extract_one_view(backbone, paths, transform, specification, device, description):
    loader = DataLoader(ImagePaths(paths, transform), batch_size=specification['batch_size'],
                        shuffle=False, num_workers=0, pin_memory=(device == 'cuda'))
    rows = []
    start = time.perf_counter()
    assert not backbone.training and not any(parameter.requires_grad for parameter in backbone.parameters())
    for index, images in enumerate(loader):
        features = backbone(images.to(device)).float().cpu()
        assert features.shape == (len(images), specification['dimension']) and torch.isfinite(features).all()
        rows.append(features)
        if index % 20 == 0 or index + 1 == len(loader):
            elapsed = time.perf_counter() - start
            remaining = elapsed / (index + 1) * (len(loader) - index - 1)
            print(f"{description}: batch {index + 1}/{len(loader)}, "
                  f"elapsed {elapsed / 60:.1f} min, remaining ~{remaining / 60:.1f} min", flush=True)
    return torch.cat(rows)


def prepare_backbone_features(branch, data, run_dir, device):
    specification = BACKBONES[branch]
    model = load_frozen_backbone(specification, device)
    fingerprint = backbone_fingerprint(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log_event(run_dir, 'backbone_loaded', branch=branch, name=specification['name'],
              frozen_parameters=parameter_count, weights_sha256=fingerprint)
    print(f"{branch}: {specification['name']}, {parameter_count:,} frozen parameters")
    evaluation, augmentation = create_transforms(specification)
    set_seed(AUGMENTATION_SEED)
    views = []
    for view in range(5):
        transform = evaluation if view == 0 else augmentation
        rows = extract_one_view(model, data['paths']['train'], transform, specification,
                                device, f'{branch} training view {view}/4')
        views.append(rows)
        log_event(run_dir, 'training_view_extracted', branch=branch, view=view, rows=len(rows))
    features = {
        'X_train': torch.cat(views),
        'X_val': extract_one_view(model, data['paths']['val'], evaluation, specification, device, f'{branch} validation'),
        'X_test': extract_one_view(model, data['paths']['test'], evaluation, specification, device, f'{branch} test'),
        # View-major order: all original images, then all images for each augmented view.
        'y_train': data['train_labels'].repeat(5), 'y_val': data['val_labels'],
    }
    assert features['X_train'].shape == (5 * len(data['train_labels']), specification['dimension'])
    assert len(features['y_train']) == len(features['X_train'])
    mean = features['X_train'].mean(0)
    std = features['X_train'].std(0, unbiased=False).clamp_min(1e-6)
    features['mean'], features['std'] = mean, std
    metadata = {'branch': branch, **specification, 'weights_sha256': fingerprint, 'hub_ref': HUB_REF,
                'augmentation_seed': AUGMENTATION_SEED, 'views': 5, 'layout': 'view-major',
                'train_transform': repr(augmentation), 'evaluation_transform': repr(evaluation)}
    # Save evidence/recovery artifacts, but never read feature caches in this notebook.
    torch.save({**features, 'metadata': metadata}, Path(run_dir) / f'{branch}_fresh_features.pt')
    log_event(run_dir, 'features_extracted', branch=branch, training_rows=len(features['X_train']),
              validation_rows=len(features['X_val']), test_rows=len(features['X_test']))
    del model, views
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, metadata


def build_classifier(kind, feature_dimension):
    if kind == 'linear':
        return nn.Linear(feature_dimension, NUM_CLASSES)
    assert kind == 'mlp'
    return nn.Sequential(
        nn.Linear(feature_dimension, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(.3),
        nn.Linear(128, NUM_CLASSES),
    )


def classification_metrics(target, probabilities):
    prediction = probabilities.argmax(1).numpy()
    return {'accuracy': float(accuracy_score(target, prediction)),
            'macro_f1': float(f1_score(target, prediction, labels=list(range(NUM_CLASSES)),
                                     average='macro', zero_division=0)),
            'correct': int((prediction == np.asarray(target)).sum()), 'count': len(target)}


@torch.inference_mode()
def predict_head(head, raw_features, mean, std, device):
    head.eval()
    probabilities = head(((raw_features - mean) / std).to(device)).softmax(1).cpu()
    assert probabilities.shape == (len(raw_features), NUM_CLASSES) and torch.isfinite(probabilities).all()
    return probabilities


def train_classifier(branch, kind, seed, features, feature_metadata, data, run_dir, device):
    rule = TRAINING_RULES[f'{branch}_{kind}']
    name = f'{branch}_{kind}_seed{seed}'
    output = Path(run_dir) / name
    output.mkdir(exist_ok=False)
    set_seed(seed)
    head = build_classifier(kind, features['X_train'].shape[1]).to(device)
    normalised_train = (features['X_train'] - features['mean']) / features['std']
    loader = DataLoader(TensorDataset(normalised_train, features['y_train']), batch_size=64,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    history = []
    best_accuracy, bad_epochs, selected_epoch = -1., 0, 0
    selected_weights = None
    optimizer_steps = 0
    start = time.perf_counter()
    log_event(run_dir, 'head_training_started', name=name, rule=rule)
    for epoch in range(1, rule['epochs'] + 1):
        head.train()
        total_loss = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x.to(device)), y.to(device))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += loss.item() * len(y)
        validation = predict_head(head, features['X_val'], features['mean'], features['std'], device)
        score = classification_metrics(features['y_val'].numpy(), validation)
        row = {'epoch': epoch, 'train_loss': total_loss / len(normalised_train), **score}
        history.append(row)
        print(f"{name} epoch {epoch:02d}: loss={row['train_loss']:.4f}, "
              f"validation accuracy={score['accuracy']:.2%}", flush=True)
        log_event(run_dir, 'head_epoch_finished', name=name, **row)
        pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
        if rule['selection'] == 'best_validation_accuracy':
            if score['accuracy'] > best_accuracy:
                best_accuracy, selected_epoch, bad_epochs = score['accuracy'], epoch, 0
                selected_weights = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            else:
                bad_epochs += 1
            if bad_epochs >= rule['patience']:
                break
        else:
            # B/L models always retain their final scheduled epoch, regardless of validation.
            selected_epoch = epoch
    if rule['selection'] == 'best_validation_accuracy':
        head.load_state_dict(selected_weights, strict=True)
    validation = predict_head(head, features['X_val'], features['mean'], features['std'], device)
    test = predict_head(head, features['X_test'], features['mean'], features['std'], device)
    score = classification_metrics(features['y_val'].numpy(), validation)
    checkpoint_path = output / 'classifier.pt'
    checkpoint = {'head_state_dict': {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                  'branch': branch, 'kind': kind, 'seed': seed, 'feature_metadata': feature_metadata,
                  'feature_mean': features['mean'], 'feature_std': features['std'],
                  'class_to_idx': data['class_to_idx'], 'training_rule': rule,
                  'selected_epoch': selected_epoch, 'epochs_run': len(history),
                  'optimizer_steps': optimizer_steps, 'run_id': Path(run_dir).name, 'metrics': score}
    with checkpoint_path.open('xb') as stream:
        torch.save(checkpoint, stream)
    record = {'branch': branch, 'kind': kind, 'seed': seed, 'selected_epoch': selected_epoch,
              'epochs_run': len(history), 'optimizer_steps': optimizer_steps,
              'training_seconds': time.perf_counter() - start, **score,
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256_file(checkpoint_path),
              'run_id': Path(run_dir).name}
    log_event(run_dir, 'head_training_finished', **record)
    np.savez_compressed(output / 'probabilities.npz', validation=validation.numpy(), test=test.numpy())
    return {'validation': validation, 'test': test, 'record': record, 'history': history}


def train_head_group(branch, kind, features, metadata, data, run_dir, device):
    return [train_classifier(branch, kind, seed, features, metadata, data, run_dir, device)
            for seed in HEAD_SEEDS]


def average_group(results, split):
    assert [result['record']['seed'] for result in results] == HEAD_SEEDS
    probabilities = torch.stack([result[split] for result in results])
    assert probabilities.ndim == 3 and probabilities.shape[0] == 3 and probabilities.shape[2] == NUM_CLASSES
    assert torch.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-5)
    return probabilities.mean(0)


def ensemble_probabilities(groups, split):
    small = average_group(groups['S_mlp'], split)
    base_mlp = average_group(groups['B_mlp'], split)
    large_mlp = average_group(groups['L_mlp'], split)
    base_linear = average_group(groups['B_linear'], split)
    large_linear = average_group(groups['L_linear'], split)
    # Soft voting: average probabilities, then choose one final class.
    base = .75 * base_mlp + .25 * base_linear
    large = .75 * large_mlp + .25 * large_linear
    return (small + base + large) / 3


def finish_run(groups, data, run_dir):
    expected_groups = {'S_mlp', 'B_mlp', 'L_mlp', 'B_linear', 'L_linear'}
    assert set(groups) == expected_groups
    records = [result['record'] for group in groups.values() for result in group]
    assert len(records) == 15 and all(record['optimizer_steps'] > 0 for record in records)
    assert all(record['run_id'] == Path(run_dir).name for record in records)
    validation = ensemble_probabilities(groups, 'validation')
    test = ensemble_probabilities(groups, 'test')
    prediction = test.argmax(1).numpy().astype(int)
    assert len(prediction) == len(data['test_ids']) and ((prediction >= 0) & (prediction < NUM_CLASSES)).all()
    output = Path(run_dir)
    pd.DataFrame({'image_id': data['test_ids'], 'label': prediction}).to_csv(output / 'submission.csv', index=False, mode='x')
    pd.DataFrame(records).to_csv(output / 'all_head_results.csv', index=False)
    group_scores = {name: classification_metrics(data['val_labels'].numpy(), average_group(results, 'validation'))
                    for name, results in groups.items()}
    final_score = classification_metrics(data['val_labels'].numpy(), validation)
    summary = {'validation': final_score, 'group_validation': group_scores, 'head_count': len(records),
               'all_heads_trained_this_run': True, 'head_records': records,
               'formula': '(S_MLP + .75*B_MLP + .25*B_linear + .75*L_MLP + .25*L_linear)/3',
               'loads_previous_classifier': False, 'loads_previous_feature_cache': False,
               'historical_reference_kaggle_score': .98376, 'this_run_kaggle_score': None,
               'submitted': False, 'test_labels_used': False}
    np.savez_compressed(output / 'ensemble_probabilities.npz', validation=validation.numpy(), test=test.numpy(),
                        validation_targets=data['val_labels'].numpy())
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    log_event(run_dir, 'ensemble_exported', head_count=15, validation=final_score, submission=str(output / 'submission.csv'))
    print('FINAL VALIDATION:', final_score)
    print('All 15 classifiers were trained in this run.')
    print('Kaggle candidate:', output / 'submission.csv')
    print('Keep this complete output folder:', output)
    return summary
