import copy,json,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import nbformat
import numpy as np
import torch
from torch import nn
from PIL import Image
import i4_ensemble_tta as m

class Encoder(nn.Module):
    def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,x):return x.flatten(1)[:,:3]+self.anchor

def checkpoint():
    return {'head_state_dict':m.build_head(3).state_dict(),'feature_mean':torch.zeros(3),'feature_std':torch.ones(3)}

class Tests(unittest.TestCase):
    def test_flip_inference_cache_and_originals(self):
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'image.png'
            Image.fromarray(np.arange(12,dtype=np.uint8).reshape(2,2,3)*20).save(image)
            transform=m.transforms.ToTensor();encoder=Encoder().eval();sources=[checkpoint(),checkpoint()]
            before=copy.deepcopy(sources);paths=[image,image]
            head=m.build_head(3);head.load_state_dict(sources[0]['head_state_dict']);head.eval()
            with torch.inference_mode():
                x=transform(Image.open(image)).flip(-1).unsqueeze(0)
                expected=head(encoder(x)).softmax(1)[0]
            meta={'branch':'synthetic','split':'test'}
            p=m.flipped_probabilities(encoder,sources,paths,transform,'cpu',1,root/'cache',meta)
            self.assertEqual(tuple(p.shape),(2,2,20));self.assertTrue(torch.allclose(p[0,0],expected,atol=1e-6))
            with patch.object(encoder,'forward',side_effect=AssertionError('cache must prevent inference')):
                q=m.flipped_probabilities(encoder,sources,paths,transform,'cpu',1,root/'cache',meta)
            self.assertTrue(torch.equal(p,q))
            for old,new in zip(before,sources):
                for k,v in old['head_state_dict'].items():self.assertTrue(torch.equal(v,new['head_state_dict'][k]))
            cache=next((root/'cache').glob('*.pt'));data=torch.load(cache,weights_only=True);data['probabilities'][0,0,0]=float('nan');torch.save(data,cache)
            with self.assertRaises(AssertionError):m.flipped_probabilities(encoder,sources,paths,transform,'cpu',1,root/'cache',meta)
    def test_probability_averaging_and_ordered_export(self):
        s,b,l=[torch.softmax(torch.randn(3,20),1) for _ in range(3)]
        fs,fb,fl=[torch.softmax(torch.randn(3,20),1) for _ in range(3)]
        result=.5*(m.equal_blend(s,b,l)+m.equal_blend(fs,fb,fl))
        expected=m.equal_blend(.5*(s+fs),.5*(b+fb),.5*(l+fl))
        self.assertTrue(torch.allclose(result,expected))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'submission.csv';m.export(path,['003','001','002'],result)
            frame=m.pd.read_csv(path,dtype={'image_id':str});self.assertEqual(frame.image_id.tolist(),['003','001','002'])
            self.assertTrue(np.array_equal(frame.label.to_numpy(),result.argmax(1).numpy()))
    def test_pipeline_exports_even_when_validation_worsens(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f0=root/'f0';i1=root/'i1';i2=root/'i2'
            for path in (f0,i1,i2):path.mkdir()
            ids=['000','001','002','003'];target=np.zeros(4,dtype=int);mapping={'class':0}
            p=torch.full((4,20),.001);p[:,0]=.981
            base=checkpoint();base.update(head_config={'input_dim':768,'hidden_dim':128,'use_bn':True,'dropout':.3,'num_classes':20},resolution=336,feature_metadata={'image_size':336,'manifest':[], 'weights_sha256':'verified'},metrics=m.metrics(target,p))
            large={'backbone':'dinov2_vitl14','resolution':336,'class_to_idx':mapping,'hub_ref':m.HUB_REF,
                'feature_metadata':{'manifest':[],'weights_sha256':'verified'},'feature_mean':torch.zeros(1024),'feature_std':torch.ones(1024),'metrics':m.metrics(target,p)}
            base_path=i1/'I1_dinov2_vitb14_336_p3_e4_aug123_head2026.pt';torch.save(base,base_path)
            lp=i2/'large.pt';torch.save(large,lp)
            paths=[];rows=[]
            for seed in m.SEEDS:
                path=f0/f'small{seed}.pt';torch.save(checkpoint(),path);paths.append(path)
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
                stack.enter_context(patch.object(m,'checked_checkpoint',side_effect=lambda path,*a:base if Path(path)==base_path else checkpoint()))
                stack.enter_context(patch.object(m,'validate_source_data'))
                stack.enter_context(patch.object(m,'cached_probabilities',return_value=(p,p)))
                stack.enter_context(patch.object(m,'load_official',return_value=Encoder()))
                stack.enter_context(patch.object(m,'infer',return_value=torch.stack([p,p,p])))
                # MagicMock supplies ImageFolder length without touching real images.
                dataset=stack.enter_context(patch.object(m.datasets,'ImageFolder'))
                dataset.return_value.class_to_idx=mapping;dataset.return_value.samples=[('unused',0)]*700;dataset.return_value.__len__.return_value=700
                flipped=torch.zeros_like(p);flipped[:,1]=1.
                stack.enter_context(patch.object(m,'flipped_probabilities',side_effect=lambda encoder,checkpoints,*a,**kw:torch.stack([flipped]*len(checkpoints))))
                output,result=m.run_i4(root,f0,i1,i2,root)
            self.assertEqual(result['validation_TTA']['correct'],0)
            self.assertEqual(result['test_changed'],4)
            candidate=m.pd.read_csv(output/'I4_S_B336_L336_flip_TTA_submission.csv',dtype={'image_id':str})
            self.assertEqual(candidate.image_id.tolist(),ids)
            self.assertTrue((candidate.label==1).all())
            for name,digest in previous.items():self.assertEqual(m.sha(name),digest)

    def test_notebook_self_contained(self):
        path=Path(__file__).resolve().parent.parent/'I4_Ensemble_FlipTTA.ipynb'
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        namespace={};exec(compile(book.cells[3].source,'I4','exec'),namespace)
        self.assertTrue(callable(namespace['run_i4']))
        self.assertNotIn('from i3_finetune',book.cells[3].source)
        for cell in book.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'I4','exec')

if __name__=='__main__':unittest.main()
