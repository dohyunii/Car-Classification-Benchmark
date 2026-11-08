import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
import build_external_training_manifest as manifest

class ExternalTrainingManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.course=self.root/'course';self.audit=self.root/'audit'
        (self.source/'images'/'train').mkdir(parents=True);(self.source/'train'/'data').mkdir(parents=True)
        self.course.mkdir();self.audit.mkdir()
        self.names=[f'Brand Model {i:03d} Sedan 2012' for i in range(196)]
        for split in ('train','test'):
            card={'id':'fixture/'+split,'sha':'revision-'+split,'cardData':{'dataset_info':{'features':[
                {'name':'image','dtype':'image'}, {'name':'label','dtype':{'class_label':{'names':{str(i):n.lower() for i,n in enumerate(self.names)}}}}]}}}
            (self.source/f'{split}_source.json').write_text(json.dumps(card))
        with (self.course/'classes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['label','class_name','folder_name']);writer.writeheader()
            for i in range(20):writer.writerow({'label':i,'class_name':self.names[i],'folder_name':f'{i:03d}_'+self.names[i].replace(' ','_')})
        self.parquet=self.source/'train'/'data'/'fixture.parquet'
        # Deliberately invalid excluded labels prove values are interpreted only after take.
        pq.write_table(pa.table({'label':pa.array([0,195,99999,1,99999],type=pa.int64()),'image':pa.array([{'bytes':f'image-{i}'.encode(),'path':None} for i in range(5)],type=pa.struct([('bytes',pa.binary()),('path',pa.string())]))}),self.parquet,row_group_size=2)
        files=[{'path':'train/data/fixture.parquet','sha256':manifest.sha256(self.parquet),'repo':'fixture/train','revision':'revision-train'}]
        (self.source/'source_provenance.json').write_text(json.dumps({'sources':{s:['fixture/'+s,'revision-'+s] for s in ('train','test')},'files':files,'images':5,'labels_loaded':False}))
        index=[]
        for i in range(5):
            relative=f'train/{i:06d}.jpg';(self.source/'images'/relative).write_bytes(f'image-{i}'.encode())
            index.append({'opaque_id':relative,'parquet':'train/data/fixture.parquet','row':i})
        (self.source/'image_index.json').write_text(json.dumps(index))
        self.set_allowed([0,1,3])
    def tearDown(self):self.temp.cleanup()
    def set_allowed(self,indices):
        allowed=[];decisions=[]
        for i in range(5):
            relative=f'train/{i:06d}.jpg';digest=manifest.sha256(self.source/'images'/relative)
            row={'opaque_id':f'source_{i:06d}','external_relative_path':relative,'file_sha256':digest,
                 'decision':'allow' if i in indices else 'quarantine_heldout_overlap',
                 'component_course_train_count':0,'component_course_val_count':0 if i in indices else 1,'component_course_test_count':0}
            decisions.append(row)
            if i in indices:allowed.append({k:row[k] for k in ('opaque_id','external_relative_path','file_sha256')})
        (self.audit/'allowlist.json').write_text(json.dumps({'audit_version':'fixture-v1','external_root':str(self.source/'images'),'images':allowed}))
        (self.audit/'summary.json').write_text(json.dumps({'version':'fixture-v1','source_images':5,'allowed_source_images':len(allowed),
            'source_labels_read':False,'test_labels_recovered_or_exported':False,'trained':False}))
        with (self.audit/'source_decisions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(decisions[0]));writer.writeheader();writer.writerows(decisions)
    def run_build(self):return manifest.build(self.source,self.audit,self.course,self.root/'out')
    def test_only_allowlisted_labels_and_target_class_mapping(self):
        selected=manifest.selected_labels(self.parquet,[3,0,1])
        self.assertEqual(selected,{0:0,1:195,3:1})
        output,result=self.run_build()
        images=result['training_images']
        self.assertEqual([r['course_label'] for r in images],[0,1])
        self.assertEqual([r['source_parquet_row'] for r in images],[0,3])
        self.assertEqual(result['summary']['source_label_rows_interpreted'],3)
        self.assertEqual(result['summary']['discarded_other_class_images'],1)
        self.assertFalse(result['summary']['protected_test_labels_joined_or_exported'])
        self.assertEqual(len(result['class_mapping']),20)
        self.assertEqual(result['summary']['per_course_class']['19'],0)
        dumped=(output/'external_training_manifest.json').read_text()
        self.assertNotIn('99999',dumped)
        self.assertNotIn('train/000002.jpg',dumped)
        self.assertNotIn('train/000004.jpg',dumped)
    def test_zero_allowed_reads_no_annotation_rows(self):
        self.set_allowed([])
        with patch.object(manifest,'selected_labels',side_effect=AssertionError('No annotation rows should be read')):
            _,result=self.run_build()
        self.assertEqual(result['training_images'],[])
        self.assertEqual(result['summary']['source_label_rows_interpreted'],0)
    def test_changed_image_and_shard_fail_closed(self):
        (self.source/'images'/'train/000000.jpg').write_bytes(b'changed')
        with self.assertRaises(AssertionError):self.run_build()
        (self.source/'images'/'train/000000.jpg').write_bytes(b'image-0')
        with self.parquet.open('ab') as stream:stream.write(b'changed')
        with self.assertRaises(AssertionError):self.run_build()
    def test_inconsistent_allowlist_or_ambiguous_mapping_refused(self):
        with (self.audit/'source_decisions.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        rows[0]['decision']='quarantine_heldout_overlap'
        with (self.audit/'source_decisions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.assertRaises(AssertionError):self.run_build()
        self.set_allowed([0,1,3])
        bad=list(self.names);bad[100]=bad[0].upper()
        with self.assertRaises(AssertionError):manifest.mapping(self.course/'classes.csv',bad)
        card=json.loads((self.source/'train_source.json').read_text())
        card['cardData']['dataset_info']['features'][1]['dtype']['class_label']['names']['195']=self.names[0]
        with self.assertRaises(AssertionError):manifest.class_names(card)
    def test_index_row_cannot_transfer_an_excluded_annotation(self):
        path=self.source/'image_index.json'
        index=json.loads(path.read_text())
        index[0]['row']=2  # This row is excluded and its label intentionally invalid.
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(AssertionError,'different parquet image bytes'):
            self.run_build()

    def test_full_names_and_safe_paths(self):
        names=list(self.names);names[0]=names[0].replace('Sedan','Coupe')
        with self.assertRaises(AssertionError):manifest.mapping(self.course/'classes.csv',names)
        with self.assertRaises(AssertionError):manifest.inside(self.source,'../outside.parquet')
        self.assertEqual(manifest.normalize(' AUDI_S4   Sedan 2007 '),'audi s4 sedan 2007')

if __name__=='__main__':unittest.main()
