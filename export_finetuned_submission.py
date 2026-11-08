"""Export a measured-improvement J2 checkpoint to CSV. Never uploads to Kaggle."""
from pathlib import Path
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
import accuracy_experiments as j0
import finetune_accuracy_experiment as j2
import preprocessing_benchmark as common


def export(checkpoint_path,data_root,batch_size=8):
    path=Path(checkpoint_path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['best_epoch']>0
    assert checkpoint['best_metrics']['accuracy']>checkpoint['baseline_metrics']['accuracy'], 'No measured J2 improvement; keep starting model'
    assert checkpoint['backbone']==j0.MODEL_NAME
    data=common.load_data(data_root)
    assert (Path(data_root)/'sample_submission.csv').exists()
    assert checkpoint['class_to_idx']==data['class_to_idx']
    assert checkpoint['source_feature_metadata']['manifest']==data['manifest']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=timm.create_model(j0.MODEL_NAME,pretrained=False,num_classes=0,dynamic_img_size=True)
    model=j2.FineTuneModel(backbone,j0.build_head(),checkpoint['feature_mean'],checkpoint['feature_std']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    transform,_=j0.make_transforms(224)
    loader=DataLoader(common.ImagePaths(data['paths'][2],transform),batch_size=batch_size,shuffle=False,num_workers=0)
    predictions=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch.to(device))
            assert logits.shape[1]==20 and torch.isfinite(logits).all()
            predictions.extend(logits.argmax(1).cpu().tolist())
    assert len(predictions)==len(data['ids']) and all(isinstance(x,int) and 0<=x<20 for x in predictions)
    frame=pd.DataFrame({'image_id':data['ids'],'label':predictions})
    sample=pd.read_csv(Path(data_root)/'sample_submission.csv',dtype={'image_id':str})
    assert frame.image_id.is_unique and frame.image_id.tolist()==sample.image_id.tolist()
    output=path.parent/'submissions'
    output.mkdir(exist_ok=True)
    csv=output/'J2_dinov2_S14_last2_seed2026_submission.csv'
    frame.to_csv(csv,index=False,mode='x')
    metadata={'checkpoint':str(path.resolve()),'checkpoint_sha256':common.file_hash(path),
              'validation_metrics':checkpoint['best_metrics'],'prediction_count':len(frame),'submitted':False}
    (output/'submission_metadata.json').write_text(json.dumps(metadata,indent=2))
    return csv

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    print(export(args.checkpoint,args.data_root))
