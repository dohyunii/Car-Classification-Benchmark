import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from contextlib import ExitStack
import nbformat
import numpy as np
import torch
import i5_nine_head as m

def make_features(dim):
    x=torch.randn(3500,dim)
    return {'X_train':x,'X_val':torch.randn(200,dim),'X_test':torch.randn(7,dim),
        'y_train':torch.arange(700).remainder(20).repeat(5),'y_val':torch.arange(200).remainder(20)}

def source(features):
    dim=features['X_train'].shape[1]
    meta={'test_ids':[f'{i:03}' for i in range(7)],'image_size':336,'feature_dim':dim,
        'seed_aug':123,'n_aug':4,'include_original':True,'layout':'view-major'}
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    return {'head_state_dict':m.build_head(dim).state_dict(),'feature_mean':features['X_train'].mean(0),
        'feature_std':features['X_train'].std(0,unbiased=False).clamp_min(1e-6),'feature_metadata':meta,
        'feature_signature':signature,'class_to_idx':{'synthetic':0},'best_epoch':3,'backbone':'synthetic',
        'training_config':{'epochs':35,'batch_size':64,'learning_rate':.001,'weight_decay':.001,'patience':7,'optimizer':'AdamW'}}

class Tests(unittest.TestCase):
    def test_training_fixed_three_epochs_and_source_unchanged(self):
        features=make_features(4);base=source(features);before=copy.deepcopy(base)
        with tempfile.TemporaryDirectory() as folder:
            # Report deliberately worsening validation: duration/selection must not change.
            reports=[{'accuracy':v,'correct':int(v*200),'macro_f1':v,'count':200} for v in (1.,.5,0.)]
            with patch.object(m,'metrics',side_effect=reports):
                pv,pt,path,result=m.train_seed(42,base,features,folder,'cpu')
            self.assertEqual(result['selected_epoch'],3);self.assertEqual(result['metrics']['accuracy'],0.)
            self.assertNotIn('best_epoch',result)
            history=m.pd.read_csv(Path(folder)/'I5_synthetic_336_head42_history.csv')
            self.assertEqual(history.epoch.tolist(),[1,2,3]);self.assertEqual(tuple(pt.shape),(7,20))
            saved=torch.load(path,weights_only=True);rv,rt=m.head_probabilities(saved,features,'cpu')
            self.assertTrue(torch.equal(pv,rv));self.assertTrue(torch.equal(pt,rt))
            with self.assertRaises(AssertionError):m.train_seed(2026,base,features,folder,'cpu')
        for key,value in before['head_state_dict'].items():self.assertTrue(torch.equal(value,base['head_state_dict'][key]))
    def test_full_cache_validation_and_normalization(self):
        features=make_features(4);base=source(features);features['metadata']=base['feature_metadata']
        training=MagicMock();training.class_to_idx=base['class_to_idx'];training.targets=torch.arange(700).remainder(20).tolist();training.__len__.return_value=700
        with tempfile.TemporaryDirectory() as folder,patch.object(m.datasets,'ImageFolder',return_value=training):
            path=Path(folder)/'student_outputs_I1_ViTB336/cache_336';path.mkdir(parents=True)
            cache=path/('cache_'+base['feature_signature']+'.pt');torch.save(features,cache)
            p,_=m.full_feature_cache(base,folder,folder,features['y_val'].numpy(),base['feature_metadata']['test_ids'])
            self.assertEqual(tuple(p['X_train'].shape),(3500,4))
            features['y_train']=features['y_train'].roll(1);torch.save(features,cache)
            with self.assertRaises(AssertionError):m.full_feature_cache(base,folder,folder,features['y_val'].numpy(),base['feature_metadata']['test_ids'])
    def test_nine_head_weights(self):
        parts=[torch.softmax(torch.randn(3,7,20),-1) for _ in range(3)]
        result=m.nine_head_blend(*parts)
        self.assertTrue(torch.allclose(result,torch.cat(parts).mean(0)))
        with self.assertRaises(AssertionError):m.nine_head_blend(parts[0][:2],parts[1],parts[2])
    def test_pipeline_exports_nine_heads_preserving_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f0=root/'f0';i1=root/'i1';i2=root/'i2'
            for path in (f0,i1,i2):path.mkdir()
            ids=['000','001','002','003'];target=np.zeros(4,dtype=int);mapping={'class':0}
            p=torch.full((4,20),.4/19);p[:,0]=.6
            base=source(make_features(4));base.update(head_config={'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},resolution=336,feature_metadata={'image_size':336,'manifest':[], 'weights_sha256':'verified'},metrics=m.metrics(target,p))
            large={'best_epoch':3,'feature_signature':'mock', 'backbone':'dinov2_vitl14','resolution':336,'class_to_idx':mapping,'hub_ref':m.HUB_REF,
                'feature_metadata':{'manifest':[],'weights_sha256':'verified'},'feature_mean':torch.zeros(1024),'feature_std':torch.ones(1024),'metrics':m.metrics(target,p)}
            base_path=i1/'I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt';torch.save(base,base_path)
            lp=i2/'large.pt';torch.save(large,lp)
            paths=[];rows=[]
            for seed in m.SEEDS:
                path=f0/f'small{seed}.pt';torch.save(source(make_features(4)),path);paths.append(path)
                rows.append({'seed':seed,'checkpoint':str(path),**m.metrics(target,p)})
            paths.append(base_path)
            originals={str(path):m.sha(path) for path in paths}
            (f0/'ensemble_summary.json').write_text(json.dumps({'head_seeds':m.SEEDS,'hub_ref':m.HUB_REF,'augmentation_seed':123,'weights':[1/3]*3,'individual_validation':rows,'ensemble_validation':m.metrics(target,p),'weights_sha256':'verified'}))
            (i2/'summary.json').write_text(json.dumps({'effective_head_weights':[1/9,1/9,1/9,1/3,1/3],
                'source_checkpoint_sha256':originals,'new_checkpoint':str(lp),'new_checkpoint_sha256':m.sha(lp),
                'validation':{'new_equal_S_B_L_blend':m.metrics(target,p)}}))
            m.export(i2/'F0_B336_L336_equal_backbone_submission.csv',ids,p)
            previous={str(path):m.sha(path) for path in paths+[lp,i2/'F0_B336_L336_equal_backbone_submission.csv']}
            with ExitStack() as stack:
                stack.enter_context(patch.object(m.torch.cuda,'is_available',return_value=True))
                stack.enter_context(patch.object(m.torch.cuda,'get_device_name',return_value='synthetic GPU'))
                stack.enter_context(patch.object(m,'load_data',return_value=(mapping,[],target,[],ids)))
                stack.enter_context(patch.object(m,'checked_checkpoint',side_effect=lambda path,*a:base if Path(path)==base_path else source(make_features(4))))
                stack.enter_context(patch.object(m,'validate_source_data'))
                stack.enter_context(patch.object(m,'cached_probabilities',return_value=(p,p)))
                stack.enter_context(patch.object(m,'load_official',return_value=MagicMock()))
                stack.enter_context(patch.object(m,'infer',return_value=torch.stack([p,p,p])))
                # MagicMock supplies ImageFolder length without touching real images.
                dataset=stack.enter_context(patch.object(m.datasets,'ImageFolder'))
                dataset.return_value.class_to_idx=mapping;dataset.return_value.samples=[('unused',0)]*700;dataset.return_value.__len__.return_value=700
                flipped=torch.zeros_like(p);flipped[:,1]=1.
                stack.enter_context(patch.object(m,'full_feature_cache',return_value=({},Path('synthetic-cache'))))
                stack.enter_context(patch.object(m,'head_probabilities',return_value=(p,p)))
                def train(seed,source,features,output,device):
                    path=output/(source['backbone']+'_head'+str(seed)+'.pt')
                    torch.save({'seed_head':seed},path)
                    return flipped,flipped,path,{'metrics':m.metrics(target,flipped)}
                stack.enter_context(patch.object(m,'train_seed',side_effect=train))
                # Match real source model names for unique output checkpoint names.
                base['backbone']='dinov2_vitb14'
                output,result=m.run_i5(root,f0,i1,i2,root)
            self.assertEqual(result['candidate_validation']['correct'],0)
            self.assertEqual(result['test_changed'],4)
            candidate=m.pd.read_csv(output/'I5_S_B336_L336_three_seed_submission.csv',dtype={'image_id':str})
            self.assertEqual(candidate.image_id.tolist(),ids)
            self.assertTrue((candidate.label==1).all())
            self.assertEqual(len(list(output.glob('*.pt'))),9)
            for name,digest in previous.items():self.assertEqual(m.sha(name),digest)

    def test_notebook_self_contained(self):
        book=nbformat.read(Path(__file__).resolve().parent.parent/'I5_ThreeSeed_Backbone_Ensemble.ipynb',as_version=4);nbformat.validate(book)
        scope={};exec(compile(book.cells[3].source,'I5','exec'),scope)
        self.assertTrue(callable(scope['run_i5']));self.assertNotIn('from i3_finetune',book.cells[3].source)
        for cell in book.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'I5','exec')

if __name__=='__main__':unittest.main()
