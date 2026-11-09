import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import assignment_training as m


class TinyFrozenEncoder(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.embed_dim=dimension;self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,x):
        rgb=x.mean((2,3))
        return rgb.repeat(1,(self.embed_dim+2)//3)[:,:self.embed_dim]+self.anchor


def dataset(root):
    counter=0
    for split,count in [('train',2),('val',1)]:
        for label in range(20):
            folder=root/split/f'{label:03d}_class';folder.mkdir(parents=True)
            for index in range(count):
                Image.new('RGB',(2,2),(counter,counter*3%256,counter*7%256)).save(folder/f'{index}.png');counter+=1
    (root/'test').mkdir()
    ids=[f'{i:06d}' for i in range(20)]
    for identity in ids:
        Image.new('RGB',(2,2),(counter,counter*3%256,counter*7%256)).save(root/'test'/(identity+'.png'));counter+=1
    pd.DataFrame({'image_id':ids,'label':np.zeros(20,dtype=int)}).to_csv(root/'sample_submission.csv',index=False)
    return ids


def notebook_scope():
    book=nbformat.read(Path(__file__).resolve().parent.parent/'Final_Car_Classifier_Training.ipynb',as_version=4)
    scope={}
    for cell in book.cells:
        if cell.cell_type!='code' or '%pip' in cell.source:continue
        tree=ast.parse(cell.source)
        for node in tree.body:
            include=isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom))
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                include='google.colab' not in ast.unparse(node)
            if isinstance(node,ast.Assign):
                include=any(isinstance(target,ast.Name) and target.id in {'HUB_REF','HEAD_SEEDS','AUGMENTATION_SEED','NUM_CLASSES','BACKBONES','TRAINING_RULES'} for target in node.targets)
            if include:exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook','exec'),scope)
    return book,scope


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def test_complete_notebook_training_with_fake_backbones(self):
        book,scope=notebook_scope()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data_root=root/'data';ids=dataset(data_root)
            data=scope['load_course_data'](data_root,expected_counts=(40,20,20))
            scope['set_seed'](123)
            run=scope['start_run'](root,data,'cpu')
            scope.update(data=data,RUN_DIR=run,DEVICE='cpu',display=lambda _:None)
            calls=[]
            def load(spec,device):
                calls.append(spec['name']);return TinyFrozenEncoder(spec['dimension']).eval()
            scope['load_frozen_backbone']=load
            scope['create_transforms']=lambda spec:(transforms.ToTensor(),transforms.Compose([transforms.RandomHorizontalFlip(.5),transforms.ToTensor()]))
            # Execute the actual notebook's three model/training sections and final ensemble cell.
            for cell in book.cells:
                if cell.cell_type=='code' and cell.source.startswith(('S_features,','B_features,','L_features,','groups =','group_names =')):
                    exec(compile(cell.source,'notebook-training','exec'),scope)
            self.assertEqual(calls,['dinov2_vits14','dinov2_vitb14','dinov2_vitl14'])
            summary=scope['summary'];self.assertEqual(summary['head_count'],15)
            self.assertTrue(summary['all_heads_trained_this_run']);self.assertIsNone(summary['this_run_kaggle_score'])
            for record in summary['head_records']:
                self.assertGreater(record['optimizer_steps'],0)
                if record['branch']!='S':self.assertEqual(record['selected_epoch'],20 if record['kind']=='linear' else 3)
                self.assertEqual(m.sha256_file(record['checkpoint']),record['checkpoint_sha256'])
                history=pd.read_csv(Path(record['checkpoint']).parent/'training_history.csv')
                selected=history.loc[history.epoch==record['selected_epoch']].iloc[0]
                self.assertAlmostEqual(selected.accuracy,record['accuracy'])
                checkpoint=torch.load(record['checkpoint'],weights_only=True)
                self.assertEqual(checkpoint['run_id'],run.name)
            self.assertEqual(len(list(run.glob('*/classifier.pt'))),15)
            self.assertTrue((run/'all_training_curves.png').is_file())
            frame=pd.read_csv(run/'submission.csv',dtype={'image_id':str});self.assertEqual(frame.image_id.tolist(),ids)
            self.assertTrue(frame.label.between(0,19).all())
            events=[json.loads(line) for line in (run/'execution_log.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='backbone_loaded' for e in events),3)
            self.assertEqual(sum(e['event']=='training_view_extracted' for e in events),15)
            self.assertEqual(sum(e['event']=='head_training_finished' for e in events),15)
            for branch in ('S','B','L'):
                features=torch.load(run/f'{branch}_fresh_features.pt',weights_only=True)
                self.assertEqual(tuple(features['X_train'].shape),(200,m.BACKBONES[branch]['dimension']))
                self.assertTrue(torch.equal(features['y_train'],data['train_labels'].repeat(5)))
                self.assertTrue(torch.equal(features['mean'],features['X_train'].mean(0)))
                self.assertTrue(torch.equal(features['std'],features['X_train'].std(0,unbiased=False).clamp_min(1e-6)))
    def test_transforms_and_explicit_probability_formula(self):
        for branch,spec in m.BACKBONES.items():
            evaluation,augmentation=m.create_transforms(spec)
            image=Image.new('RGB',(400,300),'red')
            self.assertEqual(tuple(evaluation(image).shape),(3,spec['crop'],spec['crop']))
            self.assertEqual(tuple(augmentation(image).shape),(3,spec['crop'],spec['crop']))
        groups={name:[{'record':{'seed':seed},'test':torch.softmax(torch.randn(7,20),1)} for seed in m.HEAD_SEEDS] for name in m.TRAINING_RULES}
        result=m.ensemble_probabilities(groups,'test')
        expected=sum(x['test']/9 for x in groups['S_mlp'])
        expected+=sum(x['test']/12 for name in ('B_mlp','L_mlp') for x in groups[name])
        expected+=sum(x['test']/36 for name in ('B_linear','L_linear') for x in groups[name])
        self.assertTrue(torch.allclose(result,expected,atol=1e-7))
    def test_no_previous_student_model_or_cache_loading(self):
        source=Path(m.__file__).read_text()
        self.assertNotIn('torch.load(',source)
        self.assertNotIn('car_ensemble.pt',source)
        self.assertNotIn('student_outputs_I6',source)
        book,scope=notebook_scope();nbformat.validate(book)
        for cell in book.cells:
            self.assertIsNone(cell.get('execution_count'))
            if cell.cell_type=='code':self.assertEqual(cell.outputs,[])
            if cell.cell_type=='code' and '%pip' not in cell.source:compile(cell.source,'notebook','exec')
        self.assertEqual(scope['HEAD_SEEDS'],[42,123,2026])
    def test_duplicate_split_content_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset(root)
            train=next((root/'train').rglob('*.png'));val=next((root/'val').rglob('*.png'))
            val.write_bytes(train.read_bytes())
            with self.assertRaises(AssertionError):m.load_course_data(root,expected_counts=(40,20,20))

if __name__=='__main__':unittest.main()
