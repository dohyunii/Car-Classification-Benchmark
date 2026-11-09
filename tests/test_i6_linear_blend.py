import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import nbformat
import numpy as np
import torch
from torch import nn
import i6_linear_blend as m


def source(dim=4,model='synthetic',seed=2026):
    return {'head_state_dict':m.build_head(dim).state_dict(),'feature_mean':torch.zeros(dim),'feature_std':torch.ones(dim),
        'backbone':model,'class_to_idx':{'synthetic':0},'seed_head':seed,'seed_aug':123,'resolution':336,'hub_ref':m.HUB_REF,
        'feature_metadata':{'manifest':[]},'feature_signature':'synthetic'}


class Tests(unittest.TestCase):
    def test_linear_training_fixed_twenty_epochs_and_replay(self):
        base=source();before=copy.deepcopy(base)
        features={'X_train':torch.randn(80,4),'y_train':torch.arange(80).remainder(20),
            'X_val':torch.randn(20,4),'y_val':torch.arange(20),'X_test':torch.randn(7,4)}
        reports=[{'accuracy':1-i/19,'correct':20-i,'macro_f1':0.,'count':20} for i in range(20)]
        with tempfile.TemporaryDirectory() as folder,patch.object(m,'metrics',side_effect=reports):
            pv,pt,path,c=m.train_linear(42,base,features,folder,'cpu')
            self.assertEqual(c['selected_epoch'],20);self.assertEqual(c['metrics']['accuracy'],0.)
            history=m.pd.read_csv(Path(folder)/'I6_synthetic_linear_seed42_history.csv')
            self.assertEqual(history.epoch.tolist(),list(range(1,21)))
            saved=torch.load(path,weights_only=True);head=nn.Linear(4,20);head.load_state_dict(saved['head_state_dict'])
            rv,rt=m.linear_probabilities(head,saved,features,'cpu')
            self.assertTrue(torch.equal(pv,rv));self.assertTrue(torch.equal(pt,rt))
        for name,value in before['head_state_dict'].items():self.assertTrue(torch.equal(value,base['head_state_dict'][name]))
        self.assertTrue(torch.equal(base['feature_mean'],before['feature_mean']))
    def test_recovery_and_fixed_branch_weights(self):
        s,b,l,lb,ll=[torch.softmax(torch.randn(7,20),1) for _ in range(5)]
        current=m.equal_blend(s,b,l);recovered=m.recover_small(current,b,l)
        self.assertTrue(torch.allclose(s,recovered,atol=2e-6))
        p=m.linear_mlp_blend(recovered,b,l,lb,ll)
        self.assertTrue(torch.allclose(p,(s+.75*b+.25*lb+.75*l+.25*ll)/3,atol=2e-6))
        near_zero=torch.zeros(7,20);near_zero[:,0]=1
        self.assertTrue(torch.allclose(m.recover_small(m.equal_blend(near_zero,b,l),b,l),near_zero,atol=2e-6))
        with self.assertRaises(AssertionError):m.recover_small(current*2,b,l)
    def test_pipeline_export_even_when_validation_worsens(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'I5';run.mkdir();ids=['000','003','002'];target=np.zeros(4,dtype=int)
            p=torch.full((4,20),.47/18);p[:,0]=.27;p[:,1]=.26
            pt=p[:3].clone();old_hashes={};new_hashes={}
            for model,dim in [('dinov2_vits14',384),('dinov2_vitb14',768),('dinov2_vitl14',1024)]:
                for seed in m.SEEDS:
                    c=source(dim,model,seed);c['metrics']=m.metrics(target,p)
                    path=run/f'{model}_{seed}.pt';torch.save(c,path)
                    (old_hashes if dim==384 or seed==2026 else new_hashes)[str(path)]=m.sha(path)
            summary={'head_seeds':m.SEEDS,'effective_head_weights':[1/9]*9,'TTA':False,'backbones_frozen':True,
                'source_checkpoint_sha256':old_hashes,'new_checkpoint_sha256':new_hashes,'candidate_validation':m.metrics(target,p)}
            (run/'summary.json').write_text(json.dumps(summary))
            m.export(run/'I5_S_B336_L336_three_seed_submission.csv',ids,pt)
            np.savez_compressed(run/'probabilities.npz',targets=target,validation_candidate=p.numpy(),test_candidate=pt.numpy(),
                B336_validation_heads=torch.stack([p]*3).numpy(),B336_test_heads=torch.stack([pt]*3).numpy(),
                L336_validation_heads=torch.stack([p]*3).numpy(),L336_test_heads=torch.stack([pt]*3).numpy())
            originals={str(path):m.sha(path) for path in run.iterdir()}
            groups=m.load_i5_sources(run,{'synthetic':0})[1];self.assertEqual(len(groups['dinov2_vitl14']),3)
            with ExitStack() as stack:
                stack.enter_context(patch.object(m,'load_data',return_value=({'synthetic':0},[],target,[],ids)))
                stack.enter_context(patch.object(m,'validate_source_data'))
                stack.enter_context(patch.object(m,'full_feature_cache',return_value=({},Path('synthetic-cache'))))
                stack.enter_context(patch.object(m,'head_probabilities',return_value=(p,pt)))
                def train(seed,base,features,output,device):
                    q=torch.zeros_like(p);q[:,1]=1;qt=q[:3].clone()
                    path=output/f"linear_{base['backbone']}_{seed}.pt";torch.save({'seed':seed},path)
                    return q,qt,path,{'metrics':m.metrics(target,q)}
                stack.enter_context(patch.object(m,'train_linear',side_effect=train))
                output,result=m.run_i6(root,run,root)
            self.assertEqual(result['candidate_validation']['correct'],0);self.assertEqual(result['test_changed'],3)
            frame=m.pd.read_csv(output/'I6_S_B336_L336_MLP_linear_blend_submission.csv',dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),ids);self.assertTrue((frame.label==1).all())
            self.assertEqual(len(list(output.glob('*.pt'))),15)
            for name,digest in originals.items():self.assertEqual(m.sha(name),digest)
            first=next(run.glob('*.pt'));first.write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):m.load_i5_sources(run,{'synthetic':0})
    def test_notebook_self_contained(self):
        book=nbformat.read(Path(__file__).resolve().parent.parent/'I6_Linear_MLP_Ensemble.ipynb',as_version=4);nbformat.validate(book)
        scope={};exec(compile(book.cells[3].source,'I6','exec'),scope)
        self.assertTrue(callable(scope['run_i6']));self.assertNotIn('from i5_nine_head',book.cells[3].source)
        for cell in book.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'I6','exec')

if __name__=='__main__':unittest.main()
