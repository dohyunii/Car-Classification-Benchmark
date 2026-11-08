"""Controlled frozen-DINOv1 preprocessing experiments. No dataset downloads."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import random
import time
import platform

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

MODEL_NAME = 'vit_small_patch16_224.dino'
PIPELINE_VERSION = '1'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPECS = {'P0': (True, 0, False), 'P1': (False, 1, False),
         'P2': (False, 1, True), 'P3': (True, 4, False), 'P4': (True, 4, True)}

@dataclass(frozen=True)
class Config:
    experiment_id: str = 'P0'
    global_seed: int = 123
    augmentation_seed: int = 123
    head_seed: int = 123
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 7
    head_batch_size: int = 64
    extraction_batch_size: int = 16
    expected_counts: tuple = (700, 200, 700)

    def __post_init__(self):
        if self.experiment_id not in SPECS:
            raise ValueError(f'Unknown experiment {self.experiment_id}')
        assert self.epochs > 0 and self.patience > 0


def set_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def preprocessing(config):
    original, augmented, color = SPECS[config.experiment_id]
    deterministic = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    geometry = [transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                  interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(0.5)]
    if color:
        geometry.append(transforms.RandomApply([transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)], p=0.7))
    train = transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    settings = {'include_original': original, 'augmented_views': augmented,
                'color_jitter': color, 'deterministic': repr(deterministic),
                'augmentation': repr(train) if augmented else None,
                'layout': 'view-major: original first, then augmented passes'}
    return deterministic, train, settings


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ImagePaths(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert('RGB'))


def load_data(root, expected_counts=(700, 200, 700)):
    root = Path(root).resolve()
    train, val = [datasets.ImageFolder(root / split) for split in ('train', 'val')]
    assert len(train.classes) == 20
    assert train.class_to_idx == val.class_to_idx
    assert all(name.startswith(f'{i:03d}_') for i, name in enumerate(train.classes))
    test = sorted(p for p in (root / 'test').iterdir()
                  if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
    by_id = {p.stem: p for p in test}
    assert len(by_id) == len(test), 'Duplicate test stems'
    sample = root / 'sample_submission.csv'
    if sample.exists():
        frame = pd.read_csv(sample, dtype={'image_id': str})
        assert {'image_id','label'} <= set(frame.columns)
        ids = frame.image_id.tolist()
    else:
        ids = sorted(by_id)
    assert len(ids) == len(set(ids)) and set(ids) == set(by_id)
    paths = [[Path(p) for p, _ in train.samples], [Path(p) for p, _ in val.samples],
             [by_id[i] for i in ids]]
    assert tuple(map(len, paths)) == tuple(expected_counts), 'Unexpected split counts; set explicit expected_counts for your dataset'
    # Content hashes invalidate even same-size replacements with preserved timestamps.
    manifest = []
    split_hashes = []
    for split, files in zip(('train','val','test'), paths):
        entries = [{'path': str(p.relative_to(root)), 'sha256': file_hash(p)} for p in files]
        manifest.append({'split': split, 'files': entries})
        split_hashes.append({e['sha256'] for e in entries})
    assert not any(split_hashes[i] & split_hashes[j] for i,j in ((0,1),(0,2),(1,2))), 'Identical image content across splits'
    return {'paths': paths, 'labels': [torch.tensor(train.targets), torch.tensor(val.targets)],
            'class_to_idx': train.class_to_idx, 'ids': ids, 'manifest': manifest}


def load_backbone(device):
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device)
    model.eval()
    model.requires_grad_(False)
    assert model.num_features == 384
    return model


def model_fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def extract(model, paths, transform, batch_size, device):
    model.eval()
    assert not any(p.requires_grad for p in model.parameters()), 'Backbone must be frozen'
    loader = DataLoader(ImagePaths(paths, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0))
    features = [model(batch.to(device)).float().cpu() for batch in loader]
    result = torch.cat(features)
    assert result.shape == (len(paths), 384) and torch.isfinite(result).all()
    return result


def feature_metadata(config, data, model):
    import timm, torchvision
    _, _, settings = preprocessing(config)
    return {'pipeline_version': PIPELINE_VERSION, 'model': MODEL_NAME,
            'weights_sha256': model_fingerprint(model), 'pretrained_cfg': str(getattr(model, 'pretrained_cfg', {})),
            'experiment': config.experiment_id, 'preprocessing': settings,
            'augmentation_seed': config.augmentation_seed,
            'class_to_idx': data['class_to_idx'], 'manifest': data['manifest'],
            'test_ids': data['ids'], 'extraction_batch_size': config.extraction_batch_size,
            'versions': {'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                         'timm': timm.__version__, 'pillow': Image.__version__},
            'device': str(next(model.parameters()).device)}


def valid_features(payload, metadata, data, views):
    try:
        assert payload['metadata'] == metadata
        for key, n in zip(('X_train','X_val','X_test'),
                          (len(data['paths'][0])*views,len(data['paths'][1]),len(data['paths'][2]))):
            x = payload[key]
            assert x.dtype == torch.float32 and x.shape == (n,384) and torch.isfinite(x).all()
        assert torch.equal(payload['y_train'], data['labels'][0].repeat(views))
        assert torch.equal(payload['y_val'], data['labels'][1])
        return True
    except (KeyError, AssertionError, TypeError, AttributeError):
        return False


def get_features(config, data, model, output, device, force_reextract=False):
    deterministic, augmented, settings = preprocessing(config)
    metadata = feature_metadata(config, data, model)
    cache_id = digest(metadata)
    path = Path(output)/'cache'/f'{config.experiment_id}_{cache_id}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    views = int(settings['include_original']) + settings['augmented_views']
    if path.exists() and not force_reextract:
        try:
            cached = torch.load(path, map_location='cpu', weights_only=True)
            if valid_features(cached, metadata, data, views):
                print('Loaded features from cache:', path.name)
                return cached, cache_id
            print('Invalid cache metadata/shapes; regenerating:', path.name)
        except Exception as exc:
            print('Unreadable cache; regenerating:', type(exc).__name__)
    set_seed(config.augmentation_seed)  # Once per extraction, never once per image/view.
    args = (config.extraction_batch_size, device)
    train = []
    if settings['include_original']:
        train.append(extract(model, data['paths'][0], deterministic, *args))
    for _ in range(settings['augmented_views']):
        train.append(extract(model, data['paths'][0], augmented, *args))
    payload = {'metadata': metadata, 'X_train': torch.cat(train),
               'X_val': extract(model, data['paths'][1], deterministic, *args),
               'X_test': extract(model, data['paths'][2], deterministic, *args),
               'y_train': data['labels'][0].repeat(views), 'y_val': data['labels'][1]}
    assert valid_features(payload, metadata, data, views)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    print('Freshly extracted features:', path.name)
    return payload, cache_id


def standardize(features):
    mean = features['X_train'].mean(0)
    std = features['X_train'].std(0, unbiased=False).clamp_min(1e-6)
    return [(features[k]-mean)/std for k in ('X_train','X_val','X_test')], mean, std


def train_head(config, features, device):
    standardized, mean, std = standardize(features)
    train, val, test = standardized
    set_seed(config.head_seed)  # Immediately before creating fresh head and optimizer.
    head = nn.Linear(384,20).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(train,features['y_train']), batch_size=config.head_batch_size,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(config.head_seed))
    best, best_epoch, weights, bad, history = -1, 0, None, 0, []
    for epoch in range(1,config.epochs+1):
        head.train()
        loss_sum = 0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x.to(device)),y.to(device))
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(y)
        head.eval()
        with torch.inference_mode():
            pred = head(val.to(device)).argmax(1).cpu()
        accuracy = accuracy_score(features['y_val'],pred)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train),'val_accuracy':accuracy})
        if accuracy > best:
            best, best_epoch, bad = accuracy, epoch, 0
            weights = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= config.patience: break
    head.load_state_dict(weights)
    head.eval()
    with torch.inference_mode():
        val_pred = head(val.to(device)).argmax(1).cpu().numpy()
        test_pred = head(test.to(device)).argmax(1).cpu().numpy()
    return {'head_state_dict':weights,'feature_mean':mean,'feature_std':std,
            'history':history,'best_epoch':best_epoch,'val_accuracy':best,
            'macro_f1':f1_score(features['y_val'],val_pred,labels=list(range(20)),average='macro',zero_division=0),
            'val_pred':val_pred,'test_pred':test_pred}


def run_experiment(config, data_root, output_root='outputs', force_reextract=False,
                   make_submission=False, backbone=None):
    output = Path(output_root)
    for directory in ('cache','checkpoints','metrics','plots','submissions'):
        (output/directory).mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    set_seed(config.global_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(data_root, config.expected_counts)
    model = load_backbone(device) if backbone is None else backbone.to(device)
    features, cache_id = get_features(config,data,model,output,device,force_reextract)
    extraction_seconds = time.perf_counter()-started
    training_start = time.perf_counter()
    trained = train_head(config,features,device)
    training_seconds = time.perf_counter()-training_start
    run_id = f'{config.experiment_id}_aug{config.augmentation_seed}_head{config.head_seed}_{digest({"config":asdict(config),"cache":cache_id})[:12]}'
    checkpoint = output/'checkpoints'/f'{run_id}.pt'
    torch.save({k:trained[k] for k in ('head_state_dict','feature_mean','feature_std')} |
               {'config':asdict(config),'class_to_idx':data['class_to_idx'],'backbone':MODEL_NAME,
                'cache_id':cache_id,'feature_metadata':features['metadata']},checkpoint)
    record = asdict(config) | {'run_id':run_id,'backbone':MODEL_NAME,
        'preprocessing':features['metadata']['preprocessing'], 'original_train_images':len(data['paths'][0]),
        'train_embeddings':len(features['X_train']), 'cache_id':cache_id,'checkpoint':str(checkpoint),
        'best_epoch':trained['best_epoch'],'val_accuracy':trained['val_accuracy'],'macro_f1':trained['macro_f1'],
        'feature_preparation_seconds':extraction_seconds,'training_seconds':training_seconds,
        'total_seconds':time.perf_counter()-started,'device':str(device),'python':platform.python_version()}
    (output/'metrics'/f'{run_id}.json').write_text(json.dumps(record,indent=2))
    history = pd.DataFrame(trained['history']).assign(run_id=run_id)
    history.to_csv(output/'metrics'/f'{run_id}_history.csv',index=False)
    cm = confusion_matrix(features['y_val'],trained['val_pred'],labels=list(range(20)))
    pd.DataFrame(cm,index=data['class_to_idx'],columns=data['class_to_idx']).to_csv(output/'metrics'/f'{run_id}_confusion.csv')
    fig,axes = plt.subplots(1,2,figsize=(10,3))
    axes[0].plot(history.epoch,history.train_loss); axes[0].set(xlabel='Epoch',ylabel='Training loss')
    axes[1].plot(history.epoch,history.val_accuracy); axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
    fig.suptitle(run_id); fig.tight_layout(); fig.savefig(output/'plots'/f'{run_id}_history.png'); plt.close(fig)
    if make_submission:
        predictions = trained['test_pred']
        assert len(predictions)==len(data['ids']) and ((predictions>=0)&(predictions<20)).all()
        submission = pd.DataFrame({'image_id':data['ids'],'label':predictions})
        assert submission.image_id.tolist()==data['ids'] and submission.image_id.is_unique
        submission.to_csv(output/'submissions'/f'{run_id}.csv',index=False)
    return record


def compare_results(records, output_root='outputs'):
    """Compare explicitly selected runs; never silently pool old experiment records."""
    output = Path(output_root)
    for directory in ('metrics','plots'): (output/directory).mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(records)
    assert not frame.empty and frame.run_id.is_unique, 'Pass distinct runs only'
    comparison_id = digest(sorted(frame.run_id.tolist()))[:12]
    frame.to_csv(output/'metrics'/f'comparison_{comparison_id}.csv',index=False)
    summary = frame.groupby('experiment_id').agg(n=('run_id','size'),
        accuracy_mean=('val_accuracy','mean'),accuracy_std=('val_accuracy','std'),
        f1_mean=('macro_f1','mean'),f1_std=('macro_f1','std'))
    summary.to_csv(output/'metrics'/f'summary_{comparison_id}.csv')
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    labels = frame.experiment_id+' a'+frame.augmentation_seed.astype(str)+' h'+frame.head_seed.astype(str)
    for ax,metric in zip(axes,('val_accuracy','macro_f1')):
        ax.bar(labels,frame[metric]); ax.set(ylabel=metric,ylim=(0,1)); ax.tick_params(axis='x',rotation=60)
    fig.suptitle(f'Validation runs {comparison_id} (augmentation/head seeds shown)')
    fig.tight_layout(); fig.savefig(output/'plots'/f'comparison_{comparison_id}.png'); plt.close(fig)
    if (summary.n>1).any():
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        for ax,prefix in zip(axes,('accuracy','f1')):
            ax.bar(summary.index,summary[prefix+'_mean'],yerr=summary[prefix+'_std'].fillna(0),capsize=4)
            ax.set(ylabel=f'Validation {prefix}: mean ± sample SD')
        fig.suptitle('Seeds: '+', '.join(labels)); fig.tight_layout()
        fig.savefig(output/'plots'/f'seed_summary_{comparison_id}.png'); plt.close(fig)
    return frame, summary


def run_benchmark(experiment_ids, data_root, output_root='outputs', seeds=(123,),
                  global_seed=123, force_reextract=False, make_submission=False):
    """Augmentation/head seeds follow each run seed, but remain separate in records."""
    set_seed(global_seed)
    model = load_backbone(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    records = []
    for experiment_id in experiment_ids:
        for seed in seeds:
            config = Config(experiment_id=experiment_id, global_seed=global_seed,
                            augmentation_seed=seed,head_seed=seed)
            records.append(run_experiment(config,data_root,output_root,force_reextract,make_submission,model))
    return compare_results(records,output_root)
