import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import SequentialSampler

import accuracy_experiments as j0
import external_data_training as training
import preprocessing_benchmark as common


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('identity', torch.tensor([1.]))
        self.seen_rows = 0

    def forward(self, batch):
        self.seen_rows += len(batch)
        return batch.mean((1, 2, 3)).unsqueeze(1).repeat(1, j0.FEATURE_DIM)


class ExternalDataTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.course, self.audit = [self.root/name for name in ('source', 'course', 'audit')]
        for path in (self.source/'images'/'train', self.source/'train'/'data', self.course, self.audit):
            path.mkdir(parents=True, exist_ok=True)
        self.classes = {f'{i:03d}_Model_{i}': i for i in range(20)}
        self.pairs = [{'course_label': i, 'course_class_name': f'Model {i}', 'course_folder': folder,
                       'source_class_index': i, 'source_class_name': f'model {i}'}
                      for folder, i in self.classes.items()]
        (self.course/'classes.csv').write_text('label,class_name,folder_name\n'+''.join(
            f"{p['course_label']},{p['course_class_name']},{p['course_folder']}\n" for p in self.pairs))
        paths, manifest = [], []
        for role, count, offset in [('train', 2, 10), ('val', 1, 50), ('test', 1, 90)]:
            folder = self.course/role; folder.mkdir()
            values = []
            for i in range(count):
                path = folder/f'{i}.png'; Image.new('RGB', (32, 32), (offset+i, 20, 30)).save(path)
                values.append(path)
            paths.append(values)
            manifest.append({'split': role, 'files': [{'path': str(path.relative_to(self.course)),
                                                       'sha256': common.file_hash(path)} for path in values]})
        self.data = {'paths': paths, 'labels': [torch.tensor([0, 1]), torch.tensor([0])],
                     'class_to_idx': self.classes, 'ids': ['0'], 'manifest': manifest}
        self.model = TinyEncoder().requires_grad_(False).eval()
        deterministic, augmented = j0.make_transforms(224)
        self.base = {'metadata': {'version': j0.VERSION, 'model': j0.MODEL_NAME, 'size': 224,
            'aug_seed': 123, 'original_views': 1, 'augmented_views': 4, 'layout': 'view-major',
            'weight_sha256': common.model_fingerprint(self.model), 'deterministic': repr(deterministic),
            'augmented': repr(augmented), 'class_to_idx': self.classes, 'manifest': manifest},
            'X_train': torch.arange(10*384, dtype=torch.float32).reshape(10, 384),
            'y_train': torch.tensor([0, 1]).repeat(5),
            'X_val': torch.full((1, 384), 1e9), 'y_val': torch.tensor([0])}
        self.base_path = self.root/f"J0_S14_224_{common.digest(self.base['metadata'])}.pt"
        torch.save(self.base, self.base_path)
        self.make_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def make_manifest(self):
        shard = self.source/'train'/'data'/'fixture.parquet'
        # Deliberately not a parquet file: training must only hash source bytes.
        shard.write_bytes(b'opaque annotation bytes; do not parse labels')
        shard_entry = {'path': 'train/data/fixture.parquet', 'sha256': common.file_hash(shard),
                       'repo': 'fixture/cars', 'revision': 'pinned-revision'}
        provenance = {'sources': {'train': ['fixture/cars', 'pinned-revision']}, 'files': [shard_entry]}
        (self.source/'source_provenance.json').write_text(json.dumps(provenance))
        for split in ('train', 'test'):
            (self.source/f'{split}_source.json').write_text(json.dumps({'fixture': split}))
        images, allowed, decisions, index = [], [], [], []
        for i in range(2):
            relative = f'train/{i}.png'; path = self.source/'images'/relative
            Image.new('RGB', (40, 40), (150+i*20, 160, 170)).save(path)
            digest = common.file_hash(path); identifier = f'source_{i:06d}'
            allowed.append({'opaque_id': identifier, 'external_relative_path': relative, 'file_sha256': digest})
            decisions.append({**allowed[-1], 'decision': 'allow', 'component_course_train_count': 0,
                              'component_course_val_count': 0, 'component_course_test_count': 0})
            index.append({'opaque_id': relative, 'parquet': shard_entry['path'], 'row': i})
            images.append({'external_id': identifier, 'external_relative_path': relative, 'image_path': str(path),
                           'image_sha256': digest, **self.pairs[i], 'source_dataset': shard_entry['repo'],
                           'source_revision': shard_entry['revision'], 'source_parquet': shard_entry['path'],
                           'source_parquet_row': i, 'source_parquet_sha256': shard_entry['sha256']})
        (self.source/'image_index.json').write_text(json.dumps(index))
        (self.audit/'allowlist.json').write_text(json.dumps({'audit_version': 'fixture-v1',
            'external_root': str(self.source/'images'), 'images': allowed}))
        (self.audit/'summary.json').write_text(json.dumps({'version': 'fixture-v1', 'source_images': 2,
            'allowed_source_images': 2, 'source_labels_read': False,
            'test_labels_recovered_or_exported': False, 'trained': False}))
        with (self.audit/'source_decisions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0])); writer.writeheader(); writer.writerows(decisions)
        self.manifest = {'version': training.MANIFEST_VERSION, 'training_images': images,
            'source_image_root': str(self.source/'images'), 'class_mapping': self.pairs,
            'source_shards_verified': {shard_entry['path']: shard_entry},
            'summary': {'eligible_target_class_images': 2, 'excluded_source_labels_interpreted': False,
                        'protected_test_labels_joined_or_exported': False, 'course_files_modified': False, 'trained': False}}
        self.manifest['source_cards_sha256'] = {split: common.file_hash(self.source/f'{split}_source.json')
                                                for split in ('train', 'test')}
        for file, key, root in [('source_provenance.json', 'source_provenance_sha256', self.source),
                               ('image_index.json', 'image_index_sha256', self.source),
                               ('allowlist.json', 'audit_allowlist_sha256', self.audit),
                               ('summary.json', 'audit_summary_sha256', self.audit),
                               ('source_decisions.csv', 'audit_decisions_sha256', self.audit),
                               ('classes.csv', 'course_classes_sha256', self.course)]:
            self.manifest[key] = common.file_hash(root/file)
        self.manifest_path = self.root/'external_training_manifest.json'
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest_sha = common.file_hash(self.manifest_path)

    def approved(self):
        return training.approved_images(self.manifest_path, self.manifest_sha, self.audit,
                                        self.source, self.course, self.data)

    def test_extracts_only_extra_rows_and_reuses_unchanged_course_features(self):
        _, paths, labels = self.approved()
        before = self.base['X_train'].clone(); disk_hash = common.file_hash(self.base_path)
        base, _, base_sha = training.load_base_cache(self.base_path, self.data)
        with patch.object(common, 'ImagePaths', wraps=common.ImagePaths) as datasets:
            extra, key, _ = training.extract_extra_features(self.model, paths, labels, base['metadata'],
                self.manifest_sha, self.root/'cache', torch.device('cpu'), batch_size=2)
            self.assertEqual(len(datasets.call_args_list), 5)
            for call in datasets.call_args_list:
                self.assertEqual(call.args[0], paths)
        self.assertEqual(self.model.seen_rows, 10)
        self.assertTrue(torch.equal(extra['y_extra'], torch.tensor([0, 1]*5)))
        combined = training.combine_features(base, extra, base_sha)
        self.assertTrue(torch.equal(combined['X_train'][:10], before))
        self.assertTrue(torch.equal(base['X_train'], before))
        self.assertIs(combined['X_val'], base['X_val'])
        self.assertEqual(common.file_hash(self.base_path), disk_hash)
        self.assertEqual(combined['X_train'].shape, (20, 384))
        # Held-out features are enormous and must not influence fit statistics.
        expected = torch.cat((before, extra['X_extra']))
        self.assertTrue(torch.equal(combined['X_train'].mean(0), expected.mean(0)))
        self.assertTrue(torch.equal(combined['X_train'].std(0, unbiased=False), expected.std(0, unbiased=False)))
        self.assertLess(float(combined['X_train'].mean()), 1e5)
        reused, cached_key, _ = training.extract_extra_features(self.model, paths, labels, base['metadata'],
            self.manifest_sha, self.root/'cache', torch.device('cpu'), batch_size=2)
        self.assertEqual(key, cached_key); self.assertEqual(self.model.seen_rows, 10)
        self.assertTrue(torch.equal(extra['X_extra'], reused['X_extra']))

    def test_wrong_provenance_image_and_manifest_hash_fail_closed(self):
        original_sha = self.manifest_sha
        self.manifest_path.write_text(self.manifest_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            training.approved_images(self.manifest_path, original_sha, self.audit, self.source, self.course, self.data)
        self.save_manifest()
        (self.source/'source_provenance.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source_provenance'):
            self.approved()
        self.make_manifest()
        Path(self.manifest['training_images'][0]['image_path']).write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'external image'):
            self.approved()

    def test_disallowed_and_course_copy_rejected(self):
        self.manifest['training_images'][0]['external_id'] = 'excluded_id'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'not allowlisted'):
            self.approved()
        self.make_manifest()
        # Even an internally consistent forged allowlist cannot add a course copy.
        source = Path(self.manifest['training_images'][0]['image_path'])
        source.write_bytes(self.data['paths'][0][0].read_bytes())
        copied_sha = common.file_hash(source)
        allow_path = self.audit/'allowlist.json'; allow = json.loads(allow_path.read_text())
        allow['images'][0]['file_sha256'] = copied_sha; allow_path.write_text(json.dumps(allow))
        decisions_path = self.audit/'source_decisions.csv'
        text = decisions_path.read_text().replace(self.manifest['training_images'][0]['image_sha256'], copied_sha)
        decisions_path.write_text(text)
        self.manifest['training_images'][0]['image_sha256'] = copied_sha
        self.manifest['audit_allowlist_sha256'] = common.file_hash(allow_path)
        self.manifest['audit_decisions_sha256'] = common.file_hash(decisions_path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'copies a course'):
            self.approved()

    def test_weight_mismatch_and_changed_base_labels_rejected(self):
        _, paths, labels = self.approved()
        metadata = dict(self.base['metadata']); metadata['weight_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'weights differ'):
            training.extract_extra_features(self.model, paths, labels, metadata, self.manifest_sha,
                                            self.root/'cache', torch.device('cpu'))
        self.base['y_train'][0] = 19; torch.save(self.base, self.base_path)
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            training.load_base_cache(self.base_path, self.data)

    def test_zero_eligible_manifest_is_valid_without_extraction(self):
        self.manifest['training_images'] = []; self.manifest['summary']['eligible_target_class_images'] = 0
        self.save_manifest()
        _, paths, labels = self.approved()
        self.assertEqual(paths, []); self.assertEqual(len(labels), 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            training.extract_extra_features(self.model, paths, labels, self.base['metadata'],
                self.manifest_sha, self.root/'cache', torch.device('cpu'))
        self.assertEqual(self.model.seen_rows, 0)

    def test_singleton_tail_merges_without_dropping_or_repeating_rows(self):
        for count, expected in [(129, [64, 65]), (128, [64, 64]), (130, [64, 64, 2]), (65, [65])]:
            sampler = training.MergeSingletonBatchSampler(SequentialSampler(range(count)), 64, False)
            batches = list(sampler)
            self.assertEqual([len(batch) for batch in batches], expected)
            self.assertEqual([row for batch in batches for row in batch], list(range(count)))
            self.assertEqual(len(sampler), len(batches))


if __name__ == '__main__':
    unittest.main()
