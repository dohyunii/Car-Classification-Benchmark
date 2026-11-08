import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from flip_tta_experiment import predict_flip_tta,comparison_metrics,save_submission

class PixelEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,x):return x[:,0,0,0,None]+self.anchor
class PixelHead(nn.Module):
    def forward(self,x):return torch.cat([x,-x,torch.full((len(x),18),-100.)],dim=1)

class TTATests(unittest.TestCase):
    def test_exact_probability_average_eval_and_labels_ignored(self):
        x=torch.zeros(2,3,2,2);x[:,0,0,0]=1;x[:,0,0,-1]=-3
        model,head=PixelEncoder().train(),PixelHead().train()
        loader=DataLoader(TensorDataset(x,torch.tensor([1000,2000])),batch_size=1)
        mean,std=torch.zeros(1),torch.ones(1)
        original,tta=predict_flip_tta(model,head,mean,std,loader,'cpu')
        expected=(head(torch.ones(2,1)).softmax(1)+head(torch.full((2,1),-3.)).softmax(1))/2
        self.assertTrue(torch.allclose(tta,expected))
        self.assertEqual(original.argmax(1).tolist(),[0,0]);self.assertEqual(tta.argmax(1).tolist(),[1,1])
        self.assertFalse(model.training);self.assertFalse(head.training);self.assertFalse(tta.requires_grad)
        self.assertTrue(torch.equal(mean,torch.zeros(1)))
        with self.assertRaises(AssertionError):
            predict_flip_tta(model,head,mean,torch.zeros(1),loader,'cpu')
    def test_paired_statistics(self):
        target=np.array([0,1,2])
        first=torch.eye(20)[torch.tensor([1,0,0])]
        second=torch.eye(20)[torch.tensor([0,1,2])]
        result=comparison_metrics(target,first,second)
        self.assertEqual(result['corrected'],3);self.assertEqual(result['regressed'],0)
        self.assertEqual(result['exact_paired_p_value'],0.25)
        self.assertEqual(result['horizontal_flip_tta']['accuracy'],1.)
    def test_portable_f0_helper_reproduction_gate(self):
        import F0_flip_tta_colab_cell as portable
        x=torch.zeros(200,3,2,2);x[:,0,0,0]=1;x[:,0,0,-1]=-3
        labels=torch.zeros(200,dtype=torch.long)
        val=TensorDataset(x,labels);val.targets=labels
        test=TensorDataset(x[:2])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                portable.run_verified_f0_flip_tta(PixelEncoder(),PixelHead(),torch.zeros(1),torch.ones(1),val,test,['002','001'],d,'cpu',expected_accuracy=.5)
            path=portable.run_verified_f0_flip_tta(PixelEncoder(),PixelHead(),torch.zeros(1),torch.ones(1),val,test,['002','001'],d,'cpu',expected_accuracy=1.)
            frame=pd.read_csv(path/'F0_flip_TTA_submission.csv',dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),['002','001'])
            self.assertEqual(frame.label.tolist(),[1,1])
    def test_submission_order_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'submission.csv'
            save_submission(p,['009','001'],torch.eye(20)[[3,4]])
            frame=pd.read_csv(p,dtype={'image_id':str})
            self.assertEqual(frame.image_id.tolist(),['009','001'])
            self.assertEqual(frame.label.tolist(),[3,4])
            self.assertEqual(list(frame.columns),['image_id','label'])
            with self.assertRaises(FileExistsError):save_submission(p,['009','001'],torch.eye(20)[[3,4]])
            with self.assertRaises(AssertionError):save_submission(Path(d)/'bad.csv',['001','001'],torch.eye(20)[[3,4]])
if __name__=='__main__':unittest.main()
