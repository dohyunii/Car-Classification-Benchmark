"""Extract only public Cars196 image bytes before any image labels are consulted."""
from pathlib import Path
import argparse
import hashlib
import json
import pyarrow.parquet as pq

SOURCES = {'train':('Multimodal-Fatima/StanfordCars_train','31847c49b43333c42e492bb31a21d3fcfcddab55'),
           'test':('Multimodal-Fatima/StanfordCars_test','44480308e3192efdfbb2ec4cec40b36feacbfbb7')}


def extract(root):
    root=Path(root)
    (root/'images').mkdir(exist_ok=True)
    index=[];files=[]
    for split in ('train','test'):
        count=0
        directory=root/'images'/split;directory.mkdir(exist_ok=True)
        paths=sorted((root/split/'data').glob('*.parquet'))
        assert len(paths)==3, f'Expected three pinned source shards for {split}'
        for path in paths:
            row=0
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            files.append({'path':str(path.relative_to(root)),'sha256':h.hexdigest(),'repo':SOURCES[split][0],'revision':SOURCES[split][1]})
            # Never load the label column in this preparation/audit stage.
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['image']):
                for image in batch.column(0).to_pylist():
                    raw=image['bytes'];assert raw, 'Source must contain embedded image bytes'
                    relative=f'{split}/{count:06d}.jpg'
                    destination=root/'images'/relative
                    if destination.exists():assert destination.read_bytes()==raw
                    else:destination.write_bytes(raw)
                    index.append({'opaque_id':relative,'parquet':str(path.relative_to(root)),'row':row})
                    count+=1;row+=1
                    if count%1000==0:print('EXTRACTED',split,count,flush=True)
        assert count=={'train':8144,'test':8041}[split], f'Unexpected source count {split}={count}'
    (root/'image_index.json').write_text(json.dumps(index))
    (root/'source_provenance.json').write_text(json.dumps({'sources':SOURCES,'files':files,'images':len(index),'labels_loaded':False},indent=2))
    print('EXTRACTION_COMPLETE',len(index),'labels_loaded=False',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();extract(args.root)
