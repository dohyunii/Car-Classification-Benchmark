"""Sequential synthetic F0 train/restore/TTA integration; no model downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
import nbformat

NOTEBOOK=Path(__file__).resolve().parents[1]/'F0_DINOv2_TTA.ipynb'

class Encoder(nn.Module):
    embed_dim=384
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(1))
    def forward(self,x):return x[:,:,::16,::16].flatten(1)[:,:384]+self.anchor

class NotebookTests(unittest.TestCase):
    def test_train_then_restore_sequentially(self):
        nb=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code' and not cell.source.startswith('%'):compile(cell.source,'notebook','exec')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';rng=np.random.default_rng(1234)
            for split,count in [('train',35),('val',10)]:
                for label in range(20):
                    folder=data/split/f'{label:03d}_car';folder.mkdir(parents=True)
                    for i in range(count):Image.fromarray(rng.integers(0,256,(28,32,3),dtype=np.uint8)).save(folder/f'{i:03d}.png')
            (data/'test').mkdir()
            for i in range(2):Image.fromarray(rng.integers(0,256,(28,32,3),dtype=np.uint8)).save(data/'test'/f'{i:03d}.png')
            (data/'sample_submission.csv').write_text('image_id,label\n001,0\n000,0\n')
            def execute(mode,checkpoint=None,expected=.98):
                scope={}
                with patch('torch.hub.load',return_value=Encoder()) as hub:
                    for cell in nb.cells:
                        if cell.cell_type!='code' or cell.source.startswith('%'):continue
                        code=cell.source
                        if code.startswith("MODE = 'train_f0'"):
                            code=code.replace("MODE = 'train_f0'",f'MODE = {mode!r}')
                            code=code.replace('DOWNLOAD_FROM_KAGGLE = True','DOWNLOAD_FROM_KAGGLE = False')
                            code=code.replace("Path('/content/car_data')",f'Path({str(data)!r})')
                            code=code.replace("STORAGE_ROOT = Path('/content')",f'STORAGE_ROOT = Path({str(root)!r})')
                            code=code.replace('35,1e-3,1e-3,7','1,1e-3,1e-3,7')
                            if checkpoint:
                                code=code.replace("Path('/content/dinov2_p3_e4_aug123_head2026.pt')",f'Path({str(checkpoint)!r})')
                            code=code.replace('EXPECTED_LOADED_ACCURACY = 0.98',f'EXPECTED_LOADED_ACCURACY = {expected!r}')
                        exec(compile(code,'F0-notebook','exec'),scope)
                    args,kwargs=hub.call_args
                    self.assertEqual(args[1],'dinov2_vits14');self.assertTrue(kwargs['pretrained'])
                return scope
            trained=execute('train_f0')
            self.assertEqual(trained['MODE'],'train_f0')
            self.assertEqual(tuple(trained['X_train'].shape),(3500,384))
            self.assertTrue(torch.equal(trained['y_train'],torch.tensor(trained['train_ds'].targets).repeat(5)))
            checkpoint=trained['checkpoint_path']
            saved=torch.load(checkpoint,weights_only=True)
            self.assertEqual(saved['head_state_dict']['0.weight'].shape,(128,384))
            restored=execute('load_checkpoint',checkpoint,trained['best_accuracy'])
            self.assertNotIn('X_train',restored)
            self.assertFalse(restored['backbone'].training)
            self.assertEqual(restored['summary']['validation']['original'],trained['summary']['validation']['original'])
            import pandas as pd
            a=pd.read_csv(trained['tta_output']/'F0_flip_TTA_submission.csv',dtype={'image_id':str})
            b=pd.read_csv(restored['tta_output']/'F0_flip_TTA_submission.csv',dtype={'image_id':str})
            self.assertTrue(a.equals(b));self.assertEqual(a.image_id.tolist(),['001','000'])
            self.assertFalse(restored['summary']['submitted'])
            self.assertFalse(restored['tta_output']==trained['tta_output'])

if __name__=='__main__':unittest.main()
