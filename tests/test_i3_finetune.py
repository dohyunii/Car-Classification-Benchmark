import copy, hashlib, json, tempfile, unittest
from contextlib import ExitStack
from pathlib import Path
import nbformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch
import i3_finetune as m

class Encoder(nn.Module):
    embed_dim=768
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(12)]);self.norm=nn.LayerNorm(768)
    def forward(self,x):
        for block in self.blocks:x=block(x)
        return self.norm(x.expand(-1,768))

def source():
    return {'head_state_dict':m.build_head(768).state_dict(),'feature_mean':torch.zeros(768),'feature_std':torch.ones(768)}

class Tests(unittest.TestCase):
    def test_train_modes_and_gradients(self):
        model=m.FineTuneB(Encoder(),source());model.train()
        self.assertFalse(model.backbone.training);self.assertTrue(model.head[3].training);self.assertFalse(model.head[1].training)
        before=copy.deepcopy(model.state_dict());model(torch.randn(4,1)).sum().backward()
        self.assertTrue(all(p.grad is None for block in model.backbone.blocks[:10] for p in block.parameters()))
        self.assertTrue(all(p.grad is not None for block in model.backbone.blocks[-2:] for p in block.parameters()))
        self.assertTrue(torch.equal(before['head.1.running_mean'],model.head[1].running_mean))
    def test_epoch_zero_and_improvement_selection(self):
        for improved in (False,True):
            model=m.FineTuneB(Encoder(),source());before=copy.deepcopy(model.state_dict())
            target=np.arange(4);p=torch.full((4,20),.001);p[:,0]=.981
            q=torch.full((4,20),.001);q[range(4),target]=.981
            values=[p,q if improved else p,q if improved else p]
            loader=DataLoader(TensorDataset(torch.randn(4,1),torch.tensor(target)),batch_size=4)
            with tempfile.TemporaryDirectory() as folder,patch.object(m,'image_probabilities',side_effect=values):
                epoch,score,prob,state=m.fine_tune(model,loader,[],target,None,'cpu',folder,epochs=1)
            self.assertEqual(epoch,int(improved));self.assertEqual(score['correct'],4 if improved else 1)
            if not improved:
                for key in before:self.assertTrue(torch.equal(before[key],model.state_dict()[key]),key)
    def test_cache_replay_and_rejection(self):
        target=np.arange(4);ids=['a','b','c'];base=source()
        meta={'test_ids':ids,'image_size':336};sig=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        base.update(feature_metadata=meta,feature_signature=sig)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'student_outputs_I2_ViTL336/cache_336';path.mkdir(parents=True)
            data={'metadata':meta,'y_val':torch.tensor(target),'X_val':torch.randn(4,768),'X_test':torch.randn(3,768)}
            cache=path/('cache_'+sig+'.pt');torch.save(data,cache)
            pv,pt=m.cached_probabilities(base,folder,target,ids)
            self.assertEqual(tuple(pt.shape),(3,20));self.assertTrue(torch.allclose(pv.sum(1),torch.ones(4)))
            data['X_test'][0,0]=float('nan');torch.save(data,cache)
            with self.assertRaises(AssertionError):m.cached_probabilities(base,folder,target,ids)
    def test_fixed_blend(self):
        p=torch.softmax(torch.randn(3,20),1);q=torch.softmax(torch.randn(3,20),1);r=torch.softmax(torch.randn(3,20),1)
        self.assertTrue(torch.allclose(m.equal_blend(p,q,r),(p+q+r)/3))
        with self.assertRaises(AssertionError):m.equal_blend(p*2,q,r)
    def test_pipeline_preserves_winner_and_exports_fixed_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f0=root/'f0';i1=root/'i1';i2=root/'i2'
            for path in (f0,i1,i2):path.mkdir()
            ids=['000','001','002','003'];target=np.zeros(4,dtype=int);mapping={'class':0}
            p=torch.full((4,20),.001);p[:,0]=.981
            base=source();base.update(head_config={'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},resolution=336,feature_metadata={'image_size':336,'manifest':[], 'weights_sha256':'verified'},metrics=m.metrics(target,p))
            large={'backbone':'dinov2_vitl14','resolution':336,'class_to_idx':mapping,'hub_ref':m.HUB_REF,
                'feature_metadata':{'manifest':[]},'feature_mean':torch.zeros(1024),'feature_std':torch.ones(1024),'metrics':m.metrics(target,p)}
            base_path=i1/'I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt';torch.save(base,base_path)
            lp=i2/'large.pt';torch.save(large,lp)
            paths=[];rows=[]
            for seed in m.SEEDS:
                path=f0/f'small{seed}.pt';torch.save(source(),path);paths.append(path)
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
                stack.enter_context(patch.object(m,'checked_checkpoint',side_effect=lambda path,*a:base if Path(path)==base_path else source()))
                stack.enter_context(patch.object(m,'validate_source_data'))
                stack.enter_context(patch.object(m,'cached_probabilities',return_value=(p,p)))
                stack.enter_context(patch.object(m,'load_official',return_value=Encoder()))
                stack.enter_context(patch.object(m,'infer',return_value=torch.stack([p,p,p])))
                # MagicMock supplies ImageFolder length without touching real images.
                dataset=stack.enter_context(patch.object(m.datasets,'ImageFolder'))
                dataset.return_value.class_to_idx=mapping;dataset.return_value.samples=[('unused',0)]*700;dataset.return_value.__len__.return_value=700
                stack.enter_context(patch.object(m.FineTuneB,'to',lambda self,*a,**kw:self))
                stack.enter_context(patch.object(m,'image_probabilities',return_value=p))
                stack.enter_context(patch.object(m,'fine_tune',return_value=(1,m.metrics(target,p),p,{'synthetic':torch.ones(1)})))
                output,result=m.run_i3(root,f0,i1,i2,root)
            self.assertTrue(result['candidate_exported'])
            candidate=m.pd.read_csv(output/'F0_I3_B336_L336_equal_backbone_submission.csv',dtype={'image_id':str})
            self.assertEqual(candidate.image_id.tolist(),ids)
            self.assertTrue((candidate.label==0).all())
            self.assertTrue((output/'I3_B336_finetuned.pt').is_file())
            for name,digest in previous.items():self.assertEqual(m.sha(name),digest)

    def test_notebook_self_contained(self):
        book=nbformat.read(Path(__file__).resolve().parent.parent/'I3_B336_FineTune_Ensemble.ipynb',as_version=4);nbformat.validate(book)
        namespace={}
        for i,cell in enumerate(book.cells):
            if cell.cell_type=='code' and i==3:
                self.assertNotIn('from i3_finetune',cell.source)
                exec(compile(cell.source,'notebook','exec'),namespace)
        self.assertTrue(callable(namespace['run_i3']))
        namespace['FineTuneB'](Encoder(),source())

if __name__=='__main__':unittest.main()
