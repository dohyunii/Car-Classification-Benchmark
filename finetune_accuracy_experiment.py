"""J2: bounded supervised final-two-block fine-tuning of a verified J0 model.

Only provided training images update parameters. Saved training-derived feature
statistics remain fixed. Validation selects strictly higher accuracy; epoch zero
is a real candidate and is retained whenever fine-tuning does not improve it.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import timm

import accuracy_experiments as j0
import preprocessing_benchmark as common

SEED = 2026
BACKBONE_LR = 2e-6
HEAD_LR = 3e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
MAX_EPOCHS = 10
PATIENCE = 3
CLIP_NORM = 1.0
VERSION = 'J2-v1'


class LabeledImages(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(self.labels[index])


class FineTuneModel(nn.Module):
    def __init__(self, backbone, head, mean, std):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer('feature_mean',mean.detach().clone())
        self.register_buffer('feature_std',std.detach().clone())
        assert mean.shape == std.shape == (384,)
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
        assert len(backbone.blocks) == 12
        backbone.requires_grad_(False)
        for index in (10,11): backbone.blocks[index].requires_grad_(True)
        backbone.norm.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen-backbone evaluation behavior is retained even with autograd on.
        self.backbone.eval()
        self.head.train(mode)
        for module in self.head.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def forward(self, images):
        features = self.backbone(images)
        return self.head((features-self.feature_mean)/self.feature_std)


def load_source_checkpoint(path, data):
    path = Path(path)
    source = torch.load(path,map_location='cpu',weights_only=True)
    record, meta = source['record'], source['feature_metadata']
    assert record['resolution'] == 224 and record['head_seed'] == 2026
    assert record['augmentation_seed'] == 123 and record['model'] == j0.MODEL_NAME
    assert source['class_to_idx'] == data['class_to_idx']
    assert meta['model'] == j0.MODEL_NAME and meta['size'] == 224
    assert meta['manifest'] == data['manifest']
    assert meta['class_to_idx'] == data['class_to_idx']
    assert meta['layout'] == 'view-major' and meta['original_views'] == 1 and meta['augmented_views'] == 4
    assert meta['aug_seed'] == 123
    assert common.digest(meta) == record['cache_id']
    assert round(record['accuracy']*len(data['paths'][1])) == record['correct']
    for key in ('feature_mean','feature_std'):
        assert source[key].shape == (384,) and torch.isfinite(source[key]).all()
    assert (source['feature_std'] > 0).all()
    return source


def build_model(source,device,backbone=None):
    if backbone is None:
        backbone = timm.create_model(j0.MODEL_NAME,pretrained=True,num_classes=0,dynamic_img_size=True)
    assert common.model_fingerprint(backbone) == source['feature_metadata']['weight_sha256'], 'Pretrained weights differ from source J0'
    head = j0.build_head()
    head.load_state_dict(source['head_state_dict'],strict=True)
    return FineTuneModel(backbone,head,source['feature_mean'],source['feature_std']).to(device)


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval()
    predictions,targets,probabilities = [],[],[]
    for images,labels in loader:
        logits = model(images.to(device)).float().cpu()
        assert logits.shape == (len(labels),20) and torch.isfinite(logits).all()
        probabilities.append(logits.softmax(1))
        predictions.append(logits.argmax(1)); targets.append(labels.cpu())
    predicted,actual = torch.cat(predictions),torch.cat(targets)
    metrics = {'accuracy':float(common.accuracy_score(actual,predicted)),
               'macro_f1':float(common.f1_score(actual,predicted,labels=list(range(20)),average='macro',zero_division=0)),
               'correct':int((predicted==actual).sum()),'validation_count':len(actual)}
    return metrics, predicted.numpy(), torch.cat(probabilities).numpy()


def cpu_state(model):
    return {name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}


def verify_baseline(model,loader,source,device):
    measured, predicted, probabilities = evaluate(model,loader,device)
    assert measured['correct'] == source['record']['correct'], 'J2 epoch-zero gate failed: J0 correct count was not reproduced'
    assert abs(measured['accuracy']-source['record']['accuracy']) < 1e-12, 'J2 baseline accuracy differs from J0'
    return measured,predicted,probabilities


def train_finetune(model,train_loader,val_loader,source,data,source_path,output,device,epochs=MAX_EPOCHS):
    assert 1 <= epochs <= MAX_EPOCHS, 'J2 allows at most ten predeclared epochs'
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    baseline,baseline_predictions,baseline_probabilities = verify_baseline(model,val_loader,source,device)
    common.set_seed(SEED)
    source_sha = common.file_hash(source_path)
    config = {'version':VERSION,'seed':SEED,'epochs':epochs,'patience':PATIENCE,'batch_size':BATCH_SIZE,
              'backbone_lr':BACKBONE_LR,'head_lr':HEAD_LR,'weight_decay':WEIGHT_DECAY,'clip_norm':CLIP_NORM,
              'unfrozen_blocks':[10,11],'final_norm_trainable':True,'backbone_mode':'eval with autograd',
              'head_batchnorm_running_stats':'fixed; affine trainable','head_dropout':'train',
              'normalization':'fixed saved J0 training-only mean/std','selection':'strictly higher validation accuracy; epoch zero eligible',
              'source_checkpoint':str(Path(source_path).resolve()),'source_sha256':source_sha,
              'source_weight_sha256':source['feature_metadata']['weight_sha256'],
              'environment':{'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                             'timm':timm.__version__,'pillow':Image.__version__,'python':platform.python_version(),
                             'device':str(device)}}
    counts = {'trainable_backbone':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
              'frozen_backbone':sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad),
              'trainable_head':sum(p.numel() for p in model.head.parameters() if p.requires_grad)}
    assert counts['trainable_backbone']>0 and counts['frozen_backbone']>0
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':BACKBONE_LR},
        {'params':[p for p in model.head.parameters() if p.requires_grad],'lr':HEAD_LR}],weight_decay=WEIGHT_DECAY)
    history = [{'epoch':0,'train_loss':None,**baseline}]
    best_epoch,best_metrics,bad = 0,dict(baseline),0
    best_state = cpu_state(model)
    best_predictions,best_probabilities = baseline_predictions,baseline_probabilities
    (output/'J2_config.json').write_text(json.dumps(config,indent=2))
    def save_best():
        torch.save({'model_state_dict':best_state,'feature_mean':source['feature_mean'],
                    'feature_std':source['feature_std'],'class_to_idx':data['class_to_idx'],
                    'source_feature_metadata':source['feature_metadata'],'config':config,'parameter_counts':counts,
                    'baseline_metrics':baseline,'best_metrics':best_metrics,'best_epoch':best_epoch,
                    'backbone':j0.MODEL_NAME,'head_config':j0.HEAD_CONFIG},output/'J2_best.pt')
    save_best()
    print('J2_BASELINE',json.dumps(baseline),'COUNTS',json.dumps(counts),flush=True)
    start = time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum = 0.
        for index,(images,labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = nn.functional.cross_entropy(logits,labels.to(device))
            assert torch.isfinite(loss), 'Nonfinite fine-tuning loss'
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CLIP_NORM)
            optimizer.step();loss_sum += loss.item()*len(labels)
            if index%40==0:
                print('J2_BATCH',epoch,index+1,'/',len(train_loader),'loss',loss.item(),flush=True)
        metrics,predicted,probabilities = evaluate(model,val_loader,device)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train_loader.dataset),**metrics})
        print('J2_EPOCH',epoch,json.dumps(metrics),'train_loss',history[-1]['train_loss'],flush=True)
        if metrics['accuracy'] > best_metrics['accuracy']:
            best_epoch,best_metrics,bad = epoch,dict(metrics),0
            best_state = cpu_state(model)
            best_predictions,best_probabilities = predicted,probabilities
            save_best()
        else:
            bad += 1
        pd.DataFrame(history).to_csv(output/'J2_history.csv',index=False)
        if bad >= PATIENCE: break
    model.load_state_dict(best_state,strict=True)
    restored,_,_ = evaluate(model,val_loader,device)
    assert restored['correct']==best_metrics['correct'] and restored['accuracy']==best_metrics['accuracy']
    assert not model.backbone.training
    pd.DataFrame({'path':[str(p) for p in data['paths'][1]],'true_label':data['labels'][1].numpy(),
                  'predicted_label':best_predictions,'confidence':best_probabilities.max(1)}).to_csv(
                        output/'J2_validation_predictions.csv',index=False)
    summary = {'baseline':baseline,'best':best_metrics,'best_epoch':best_epoch,'parameter_counts':counts,
               'improved':best_epoch>0,'epochs_run':len(history)-1,'elapsed_seconds':time.perf_counter()-start,
               'checkpoint':str(output/'J2_best.pt'),'no_test_inference':True,'no_kaggle_submission':True}
    (output/'J2_summary.json').write_text(json.dumps(summary,indent=2))
    print('J2_COMPLETE',json.dumps(summary),flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',default='outputs/J2')
    parser.add_argument('--epochs',type=int,default=MAX_EPOCHS,help='Default10; lower values for smoke tests only')
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    common.set_seed(SEED)
    data = common.load_data(args.data_root)
    source = load_source_checkpoint(args.source_checkpoint,data)
    assert source['record']['correct']==195 and source['record']['accuracy']==0.975, 'J2 requires verified J0 195/200 source'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(source,device)
    deterministic,augmented = j0.make_transforms(224)
    train_loader = DataLoader(LabeledImages(data['paths'][0],data['labels'][0],augmented),batch_size=BATCH_SIZE,
                shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(LabeledImages(data['paths'][1],data['labels'][1],deterministic),batch_size=BATCH_SIZE,
                           shuffle=False,num_workers=0)
    output = Path(args.output_root)/uuid.uuid4().hex[:12]
    print('J2_OUTPUT',output.resolve(),flush=True)
    train_finetune(model,train_loader,val_loader,source,data,args.source_checkpoint,output,device,args.epochs)


if __name__=='__main__': main()
