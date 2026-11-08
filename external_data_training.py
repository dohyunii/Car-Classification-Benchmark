"""Train frozen DINOv2-S heads using only completed-audit Cars196 additions.

The J0 course features are immutable inputs. This command never extracts course
features, reads source annotations, predicts the course test split, or submits.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import uuid

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset
import timm

import accuracy_experiments as j0
import preprocessing_benchmark as common

VERSION = 'Cars196-external-training-v1'
MANIFEST_VERSION = 'Cars196-allowlisted-training-manifest-v1'
SIZE = 224


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path != root and path.is_relative_to(root), 'Source path escapes root')
    return path


def verify_hash(path, expected, description):
    require(isinstance(expected, str) and len(expected) == 64,
            f'Missing SHA256 for {description}')
    require(Path(path).is_file() and common.file_hash(path) == expected,
            f'{description} changed or has the wrong SHA256')


def approved_images(manifest_path, manifest_sha256, audit_dir, source_root, course_root, data):
    """Verify approvals without opening any source annotation or source-label column."""
    manifest_path, audit_dir = Path(manifest_path), Path(audit_dir)
    source_root, course_root = Path(source_root).resolve(), Path(course_root).resolve()
    verify_hash(manifest_path, manifest_sha256, 'Approved training manifest')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['version'] == MANIFEST_VERSION, 'Unsupported training manifest version')
    for relative, key in [('allowlist.json', 'audit_allowlist_sha256'),
                          ('summary.json', 'audit_summary_sha256'),
                          ('source_decisions.csv', 'audit_decisions_sha256')]:
        verify_hash(audit_dir / relative, manifest[key], f'Audit {relative}')
    for relative, key in [('source_provenance.json', 'source_provenance_sha256'),
                          ('image_index.json', 'image_index_sha256')]:
        verify_hash(source_root / relative, manifest[key], relative)
    verify_hash(course_root / 'classes.csv', manifest['course_classes_sha256'], 'Course class mapping')
    for split in ('train', 'test'):
        verify_hash(source_root / f'{split}_source.json', manifest['source_cards_sha256'][split],
                    f'{split} source card')
    allow = json.loads((audit_dir / 'allowlist.json').read_text())
    audit = json.loads((audit_dir / 'summary.json').read_text())
    require(allow['audit_version'] == audit['version'], 'Audit version mismatch')
    require(audit['source_labels_read'] is False and audit['test_labels_recovered_or_exported'] is False
            and audit['trained'] is False, 'Audit was not label blind')
    image_root = (source_root / 'images').resolve()
    require(Path(allow['external_root']).resolve() == image_root == Path(manifest['source_image_root']).resolve(),
            'Source image root differs from audited root')
    require(not image_root.is_relative_to(course_root), 'External root is inside the course dataset')
    allowed = {row['opaque_id']: row for row in allow['images']}
    require(len(allowed) == len(allow['images']) == audit['allowed_source_images'], 'Invalid allowlist count')
    with (audit_dir / 'source_decisions.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    decisions = {row['opaque_id']: row for row in rows}
    require(len(decisions) == len(rows), 'Duplicate source decisions')
    provenance = json.loads((source_root / 'source_provenance.json').read_text())
    source_files = {row['path']: row for row in provenance['files']}
    require(len(source_files) == len(provenance['files']), 'Duplicate provenance paths')
    index_rows = json.loads((source_root / 'image_index.json').read_text())
    index = {row['opaque_id']: row for row in index_rows}
    require(len(index) == len(index_rows) == audit['source_images'], 'Invalid image index')
    # Hash bytes only; never decode labels from the parquet shards.
    for relative, entry in manifest['source_shards_verified'].items():
        require(entry == source_files.get(relative), 'Shard provenance does not match source snapshot')
        verify_hash(safe_path(source_root, relative), entry['sha256'], 'Verified source shard')
    mapping = {row['course_label']: row for row in manifest['class_mapping']}
    require(set(mapping) == set(range(20)), 'Expected exactly 20 course classes')
    require(len({row['source_class_index'] for row in mapping.values()}) == 20, 'Ambiguous source classes')
    for label, pair in mapping.items():
        require(data['class_to_idx'].get(pair['course_folder']) == label, 'Course class mapping differs')
    summary = manifest['summary']
    require(summary['excluded_source_labels_interpreted'] is False
            and summary['protected_test_labels_joined_or_exported'] is False
            and summary['course_files_modified'] is False and summary['trained'] is False,
            'Manifest violates split or source-label restrictions')
    entries = manifest['training_images']
    require(len(entries) == summary['eligible_target_class_images'], 'Training image count differs')
    course_paths = {Path(path).resolve() for paths in data['paths'] for path in paths}
    course_hashes = {entry['sha256'] for split in data['manifest'] for entry in split['files']}
    paths, labels, seen_ids, seen_paths, seen_hashes = [], [], set(), set(), set()
    for row in entries:
        identifier, relative = row['external_id'], row['external_relative_path']
        approved, decision = allowed.get(identifier), decisions.get(identifier)
        require(approved is not None and decision is not None, 'Training image is not allowlisted')
        require(approved['external_relative_path'] == relative == decision['external_relative_path']
                and approved['file_sha256'] == row['image_sha256'] == decision['file_sha256'],
                'Training image differs from approved image')
        require(decision['decision'] == 'allow' and all(int(decision[key]) == 0 for key in
                ('component_course_train_count', 'component_course_val_count', 'component_course_test_count')),
                'Training image belongs to an excluded or protected component')
        path = safe_path(image_root, relative)
        require(path == Path(row['image_path']).resolve(), 'Training image path differs from audit')
        require(path not in course_paths and not path.is_relative_to(course_root), 'Course image used as external data')
        verify_hash(path, row['image_sha256'], 'Approved external image')
        require(row['image_sha256'] not in course_hashes, 'External image copies a course image')
        require(identifier not in seen_ids and path not in seen_paths and row['image_sha256'] not in seen_hashes,
                'Duplicate external image')
        seen_ids.add(identifier); seen_paths.add(path); seen_hashes.add(row['image_sha256'])
        pair = mapping.get(row['course_label'])
        require(pair is not None and all(row[key] == value for key, value in pair.items()),
                'Training label differs from approved class mapping')
        item = index.get(relative)
        require(item is not None and item['parquet'] == row['source_parquet']
                and item['row'] == row['source_parquet_row'], 'Image-to-source-row provenance differs')
        shard = manifest['source_shards_verified'].get(row['source_parquet'])
        require(shard is not None and shard['sha256'] == row['source_parquet_sha256']
                and shard['repo'] == row['source_dataset'] and shard['revision'] == row['source_revision'],
                'Training source revision differs from verified shard')
        paths.append(path); labels.append(row['course_label'])
    return manifest, paths, torch.tensor(labels, dtype=torch.long)


def load_base_cache(path, data):
    """Load the existing raw P3 features; never rebuild or alter its views."""
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    deterministic, augmented = j0.make_transforms(SIZE)
    require(metadata['version'] == j0.VERSION and metadata['model'] == j0.MODEL_NAME
            and metadata['size'] == SIZE and metadata['aug_seed'] == j0.AUG_SEED
            and metadata['original_views'] == 1 and metadata['augmented_views'] == 4
            and metadata['layout'] == 'view-major', 'Base cache is not the matched J0 P3 cache')
    require(metadata['deterministic'] == repr(deterministic) and metadata['augmented'] == repr(augmented),
            'Base preprocessing differs from current P3 transforms')
    require(metadata['manifest'] == data['manifest'] and metadata['class_to_idx'] == data['class_to_idx'],
            'Course dataset changed since the base cache was extracted')
    for key, count in [('X_train', len(data['paths'][0])*5), ('X_val', len(data['paths'][1]))]:
        value = payload[key]
        require(value.dtype == torch.float32 and value.shape == (count, j0.FEATURE_DIM)
                and torch.isfinite(value).all(), 'Invalid raw base features')
    require(torch.equal(payload['y_train'], data['labels'][0].repeat(5))
            and torch.equal(payload['y_val'], data['labels'][1]), 'Base cache labels differ')
    cache_id = common.digest(metadata)
    require(path.name == f'J0_S14_224_{cache_id}.pt', 'Base cache filename does not match metadata digest')
    return payload, cache_id, common.file_hash(path)


def extract_extra_features(model, paths, labels, base_metadata, manifest_sha256,
                           cache_root, device, batch_size=8):
    require(len(paths) > 0 and len(paths) == len(labels), 'Expected positive approved external data')
    require(batch_size > 0, 'Batch size must be positive')
    require(not any(parameter.requires_grad for parameter in model.parameters()), 'Backbone must be frozen')
    require(common.model_fingerprint(model) == base_metadata['weight_sha256'],
            'Backbone weights differ from the original J0 feature cache')
    deterministic, augmented = j0.make_transforms(SIZE)
    require(repr(deterministic) == base_metadata['deterministic']
            and repr(augmented) == base_metadata['augmented'], 'P3 transform mismatch')
    metadata = {'version': VERSION, 'manifest_sha256': manifest_sha256, 'model': j0.MODEL_NAME,
                'weight_sha256': base_metadata['weight_sha256'], 'size': SIZE,
                'deterministic': repr(deterministic), 'augmented': repr(augmented),
                'augmentation_seed': j0.AUG_SEED, 'original_views': 1, 'augmented_views': 4,
                'layout': 'external-view-major; append after unchanged course-view-major',
                'images': [{'path': str(path), 'sha256': common.file_hash(path), 'label': int(label)}
                           for path, label in zip(paths, labels)],
                'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'pillow': Image.__version__, 'timm': timm.__version__, 'device': str(device),
                'batch_size': batch_size, 'workers': 0}
    key = common.digest(metadata)
    path = Path(cache_root) / f'Cars196_extra_P3_{key}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    def validate(payload):
        require(payload['metadata'] == metadata, 'External feature cache metadata differs')
        value = payload['X_extra']
        require(value.dtype == torch.float32 and value.shape == (len(paths)*5, j0.FEATURE_DIM)
                and torch.isfinite(value).all(), 'External feature cache invalid')
        require(torch.equal(payload['y_extra'], labels.repeat(5)), 'External feature labels differ')
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        validate(payload)
        print('EXTRA_CACHE_HIT', str(path), flush=True)
        return payload, key, path
    common.set_seed(j0.AUG_SEED)
    model.eval()
    def extract(transform, name):
        loader = DataLoader(common.ImagePaths(paths, transform), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        values = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                values.append(model(batch.to(device)).float().cpu())
                if index % 40 == 0:
                    print('EXTRACT_EXTRA', name, min((index+1)*batch_size, len(paths)), '/', len(paths), flush=True)
        return torch.cat(values)
    views = [extract(deterministic, 'original')]
    for view in range(4):
        views.append(extract(augmented, f'aug{view+1}'))
    payload = {'metadata': metadata, 'X_extra': torch.cat(views), 'y_extra': labels.repeat(5)}
    validate(payload)
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    torch.save(payload, temporary); temporary.replace(path)
    return payload, key, path


def combine_features(base, extra, base_sha256):
    require(extra['X_extra'].shape[1:] == (j0.FEATURE_DIM,), 'Wrong external feature width')
    combined = {'X_train': torch.cat((base['X_train'], extra['X_extra'])),
                'y_train': torch.cat((base['y_train'], extra['y_extra'])),
                'X_val': base['X_val'], 'y_val': base['y_val'],
                'metadata': {'version': VERSION, 'base_cache_sha256': base_sha256,
                             'base_metadata': base['metadata'], 'external_metadata': extra['metadata'],
                             'weight_sha256': base['metadata']['weight_sha256'],
                             'base_training_rows': len(base['X_train']),
                             'external_training_rows': len(extra['X_extra']),
                             'normalization': 'combined training rows only; population std clamped at 1e-6'}}
    require(torch.equal(combined['X_train'][:len(base['X_train'])], base['X_train'])
            and torch.equal(combined['y_train'][:len(base['y_train'])], base['y_train']),
            'Original feature rows changed')
    return combined


class MergeSingletonBatchSampler(BatchSampler):
    """Keep every shuffled row once; merge a singleton tail into its preceding batch."""
    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    return
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = super().__len__()
        return count - int(len(self.sampler) > self.batch_size and len(self.sampler) % self.batch_size == 1)


def train_head(features, seed, cache_id, data, output, device):
    """Same E4/J0 loop, with a deterministic 65-row tail only when BN requires it."""
    settings = dict(j0.TRAIN_CONFIG)
    mean = features['X_train'].mean(0)
    std = features['X_train'].std(0, unbiased=False).clamp_min(1e-6)
    train = (features['X_train']-mean)/std
    val = (features['X_val']-mean)/std
    common.set_seed(seed)
    head = j0.build_head().to(device)
    dataset = TensorDataset(train, features['y_train'])
    generator = torch.Generator().manual_seed(seed)
    batches = MergeSingletonBatchSampler(RandomSampler(dataset, generator=generator), settings['batch_size'], False)
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0, generator=generator)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    best, bad, best_epoch, weights = -1., 0, 0, None
    history = []; start = time.perf_counter()
    for epoch in range(1, settings['epochs']+1):
        head.train(); loss_sum = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x.to(device)), y.to(device))
            loss.backward(); optimizer.step(); loss_sum += loss.item()*len(y)
        head.eval()
        with torch.inference_mode():
            pred = head(val.to(device)).argmax(1).cpu()
        accuracy = float(common.accuracy_score(features['y_val'], pred))
        history.append({'epoch': epoch, 'loss': loss_sum/len(train), 'val_accuracy': accuracy})
        if accuracy > best:
            best, bad, best_epoch = accuracy, 0, epoch
            weights = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            bad += 1
            if bad >= settings['patience']:
                break
    head.load_state_dict(weights); head.eval()
    with torch.inference_mode():
        logits = head(val.to(device)).cpu()
    pred = logits.argmax(1)
    name = f'Cars196_S14_224_aug123_head{seed}'
    record = {'experiment': VERSION, 'resolution': SIZE, 'head_seed': seed, 'augmentation_seed': j0.AUG_SEED,
              'accuracy': best, 'macro_f1': float(common.f1_score(features['y_val'], pred, labels=list(range(20)),
              average='macro', zero_division=0)), 'correct': int((pred == features['y_val']).sum()),
              'val_count': len(pred), 'best_epoch': best_epoch, 'epochs_run': len(history),
              'model': j0.MODEL_NAME, 'cache_id': cache_id, 'head_config': dict(j0.HEAD_CONFIG),
              'training_config': settings, 'training_rows': len(train),
              'singleton_tail_merged': len(train) % settings['batch_size'] == 1,
              'training_seconds': time.perf_counter()-start,
              'checkpoint': str(output/'checkpoints'/f'{name}.pt')}
    torch.save({'head_state_dict': weights, 'feature_mean': mean, 'feature_std': std, 'record': record,
                'class_to_idx': data['class_to_idx'], 'feature_metadata': features['metadata']}, record['checkpoint'])
    (output/'metrics'/f'{name}.json').write_text(json.dumps(record, indent=2))
    pd.DataFrame(history).to_csv(output/'metrics'/f'{name}_history.csv', index=False)
    pd.DataFrame({'path': [str(path) for path in data['paths'][1]], 'true_label': features['y_val'].numpy(),
                  'predicted_label': pred.numpy(), 'confidence': logits.softmax(1).max(1).values.numpy()}).to_csv(
                      output/'metrics'/f'{name}_validation_predictions.csv', index=False)
    return record


def baseline_records(baseline_run, base, base_cache_id, data):
    records = []
    for seed in j0.SEEDS:
        path = Path(baseline_run)/'checkpoints'/f'J0_S14_224_aug123_head{seed}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        record = checkpoint['record']
        require(checkpoint['feature_metadata'] == base['metadata']
                and checkpoint['class_to_idx'] == data['class_to_idx']
                and record['cache_id'] == base_cache_id and record['head_seed'] == seed
                and record['head_config'] == j0.HEAD_CONFIG and record['training_config'] == j0.TRAIN_CONFIG,
                'Baseline checkpoint does not match the frozen P3/E4 control')
        require(torch.equal(checkpoint['feature_mean'], base['X_train'].mean(0))
                and torch.equal(checkpoint['feature_std'], base['X_train'].std(0, unbiased=False).clamp_min(1e-6)),
                'Baseline normalization differs from course-training-only statistics')
        head = j0.build_head(); head.load_state_dict(checkpoint['head_state_dict']); head.eval()
        with torch.inference_mode():
            pred = head((base['X_val']-checkpoint['feature_mean'])/checkpoint['feature_std']).argmax(1)
        require(float(common.accuracy_score(base['y_val'], pred)) == record['accuracy'],
                'Baseline validation accuracy differs from its checkpoint')
        f1 = float(common.f1_score(base['y_val'], pred, labels=list(range(20)), average='macro', zero_division=0))
        require(abs(f1-record['macro_f1']) < 1e-12, 'Baseline validation F1 differs from checkpoint')
        records.append({**record, 'experiment': 'J0_course_only_control', 'checkpoint_sha256': common.file_hash(path)})
    return records


def write_comparison(baseline, augmented, output):
    frame = pd.DataFrame(baseline + augmented)
    frame.to_csv(output/'metrics'/'runs.csv', index=False)
    summary = frame.groupby('experiment', sort=False).agg(n=('accuracy', 'size'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'))
    summary.to_csv(output/'metrics'/'summary.csv')
    pairs = []
    for new in augmented:
        control = next(row for row in baseline if row['head_seed'] == new['head_seed'])
        pairs.append({'head_seed': new['head_seed'], 'baseline_accuracy': control['accuracy'],
                      'external_accuracy': new['accuracy'], 'accuracy_delta': new['accuracy']-control['accuracy'],
                      'baseline_macro_f1': control['macro_f1'], 'external_macro_f1': new['macro_f1'],
                      'macro_f1_delta': new['macro_f1']-control['macro_f1']})
    if pairs:
        pd.DataFrame(pairs).to_csv(output/'metrics'/'paired_deltas.csv', index=False)
    return summary.reset_index().to_dict(orient='records')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--base-cache', required=True)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--output-root', default='outputs/Cars196_training')
    parser.add_argument('--cache-root', default='outputs/Cars196_training/cache')
    parser.add_argument('--batch-size', type=int, default=8, help='Image extraction batch size; head batch size stays 64')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    data = common.load_data(args.data_root)
    base, base_id, base_sha = load_base_cache(args.base_cache, data)
    manifest, paths, labels = approved_images(args.manifest, args.manifest_sha256, args.audit_dir,
                                              args.source_root, args.data_root, data)
    baseline = baseline_records(args.baseline_run, base, base_id, data)
    output = Path(args.output_root)/uuid.uuid4().hex[:12]
    for name in ('cache', 'checkpoints', 'metrics'):
        (output/name).mkdir(parents=True, exist_ok=True)
    print('OUTPUT', str(output.resolve()), 'ELIGIBLE_IMAGES', len(paths), flush=True)
    provenance = {'manifest_path': str(Path(args.manifest).resolve()), 'manifest_sha256': args.manifest_sha256,
                  'base_cache': str(Path(args.base_cache).resolve()), 'base_cache_sha256': base_sha,
                  'base_cache_id': base_id, 'baseline_run': str(Path(args.baseline_run).resolve()),
                  'audit_dir': str(Path(args.audit_dir).resolve()), 'source_root': str(Path(args.source_root).resolve()),
                  'data_root': str(Path(args.data_root).resolve()), 'approved_manifest': manifest}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    if not paths:
        summary = {'status': 'skipped_no_eligible_external_images', 'eligible_external_images': 0,
                   'trained': False, 'test_inference_performed': False, 'kaggle_submitted': False,
                   'comparison': write_comparison(baseline, [], output)}
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = timm.create_model(j0.MODEL_NAME, pretrained=True, num_classes=0, dynamic_img_size=True).to(device)
        model.requires_grad_(False).eval()
        extra, extra_id, extra_cache = extract_extra_features(model, paths, labels, base['metadata'],
            args.manifest_sha256, args.cache_root, device, args.batch_size)
        combined = combine_features(base, extra, base_sha)
        cache_id = common.digest(combined['metadata'])
        combined_cache = output/'cache'/f'combined_{cache_id}.pt'
        torch.save(combined, combined_cache)
        torch.save({'mean': combined['X_train'].mean(0),
                    'std': combined['X_train'].std(0, unbiased=False).clamp_min(1e-6),
                    'training_rows': len(combined['X_train']), 'val_rows_used_in_fit': 0,
                    'test_rows_used_in_fit': 0, 'cache_id': cache_id}, output/'training_statistics.pt')
        records = []
        for seed in j0.SEEDS:
            record = train_head(combined, seed, cache_id, data, output, device)
            records.append(record)
            print('RESULT', json.dumps(record), flush=True)
        require(common.file_hash(args.base_cache) == base_sha, 'Original base cache changed during training')
        summary = {'status': 'completed', 'eligible_external_images': len(paths),
                   'course_train_images': len(data['paths'][0]), 'course_train_feature_rows': len(base['X_train']),
                   'external_feature_rows': len(extra['X_extra']), 'combined_feature_rows': len(combined['X_train']),
                   'val_images': len(base['X_val']), 'head_seeds': list(j0.SEEDS),
                   'trained': True, 'test_inference_performed': False, 'kaggle_submitted': False,
                   'extra_cache_id': extra_id, 'extra_cache': str(extra_cache),
                   'combined_cache_id': cache_id, 'combined_cache': str(combined_cache),
                   'comparison': write_comparison(baseline, records, output)}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('EXTERNAL_TRAINING_COMPLETE', str(output.resolve()), json.dumps(summary), flush=True)
    return output, summary


if __name__ == '__main__':
    main()
