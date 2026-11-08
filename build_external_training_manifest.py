"""Interpret source labels only for image IDs retained by the completed audit.

No images are copied, no course split changes, and no training/test predictions
are performed. Source annotation rows excluded by the image audit never become
Python label values or get joined to competition image IDs.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import numbers
import re
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

VERSION='Cars196-allowlisted-training-manifest-v1'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def inside(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    assert path.is_relative_to(root) and path!=root,'Path escapes source root'
    return path


def normalize(name):
    assert isinstance(name,str) and name.strip()
    # Case, underscores and spacing only; keep model/body/year punctuation intact.
    return re.sub(r'\s+',' ',name.replace('_',' ').strip()).casefold()


def class_names(card):
    info=card['cardData']['dataset_info']
    assert isinstance(info,dict),'Ambiguous multi-configuration source card'
    features=info['features']
    feature=[row for row in features if row.get('name')=='label']
    assert len(feature)==1,'Exactly one documented source class-label field required'
    names=feature[0]['dtype']['class_label']['names']
    if isinstance(names,dict):
        keys=sorted(int(k) for k in names)
        assert keys==list(range(196)),'Source label names must index all196 classes contiguously'
        names=[names[str(i)] for i in keys]
    assert isinstance(names,list) and len(names)==196
    assert len(set(map(normalize,names)))==196,'Duplicate/ambiguous full source class names'
    return names


def mapping(course_classes,names):
    with Path(course_classes).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==20 and sorted(int(r['label']) for r in rows)==list(range(20))
    lookups=defaultdict(list)
    for index,name in enumerate(names):lookups[normalize(name)].append(index)
    pairs=[]
    for row in sorted(rows,key=lambda r:int(r['label'])):
        matches=lookups.get(normalize(row['class_name']),[])
        assert len(matches)==1,f"No unique exact full source class-name match: {row['class_name']}"
        index=matches[0]
        pairs.append({'course_label':int(row['label']),'course_class_name':row['class_name'],
                      'course_folder':row['folder_name'],'source_class_index':index,'source_class_name':names[index]})
    assert len({p['source_class_index'] for p in pairs})==20
    return pairs


def selected_labels(path,rows,image_sha_by_row=None):
    """Take selected rows before interpreting values; never .to_pylist() a whole label column."""
    rows=list(rows)
    assert rows and len(set(rows))==len(rows) and all(isinstance(r,int) and r>=0 for r in rows)
    file=pq.ParquetFile(path)
    assert pa.types.is_integer(file.schema_arrow.field('label').type),'Source label field must be documented integer class IDs'
    assert max(rows)<file.metadata.num_rows
    groups=[];offset=0;wanted=sorted(rows);local_indices=[];selected_offset=0
    for group in range(file.metadata.num_row_groups):
        count=file.metadata.row_group(group).num_rows
        chosen=[row for row in wanted if offset<=row<offset+count]
        if chosen:
            groups.append(group)
            local_indices.extend(selected_offset+row-offset for row in chosen)
            selected_offset+=count
        offset+=count
    # Arrow may decode a selected row group's label column, but only allowed row
    # positions are converted to Python/validated/interpreted as class labels.
    table=file.read_row_groups(groups,columns=['label'] if image_sha_by_row is None else ['image','label'])
    allowed=table.take(pa.array(local_indices,type=pa.int64()))
    if image_sha_by_row is not None:
        # Validate the index-to-image join before interpreting any selected label.
        for row,image in zip(wanted,allowed.column('image').to_pylist()):
            assert isinstance(image,dict) and isinstance(image.get('bytes'),bytes)
            assert hashlib.sha256(image['bytes']).hexdigest()==image_sha_by_row[row], 'Source image index points to different parquet image bytes'
    values=allowed.column('label').to_pylist()
    assert len(values)==len(wanted)
    result={}
    for row,value in zip(wanted,values):
        assert isinstance(value,numbers.Integral) and not isinstance(value,bool) and 0<=value<196
        result[row]=int(value)
    return result


def build(source_root,audit_dir,course_root,output_root):
    source_root,audit_dir,course_root=Path(source_root),Path(audit_dir),Path(course_root)
    allow_path=audit_dir/'allowlist.json';summary_path=audit_dir/'summary.json';decisions_path=audit_dir/'source_decisions.csv'
    allow=json.loads(allow_path.read_text());audit=json.loads(summary_path.read_text())
    assert allow['audit_version']==audit['version']
    assert audit['source_labels_read'] is False and audit['test_labels_recovered_or_exported'] is False
    assert audit['trained'] is False
    assert Path(allow['external_root']).resolve()==(source_root/'images').resolve()
    allowed=allow['images']
    assert len(allowed)==audit['allowed_source_images']
    assert len({r['opaque_id'] for r in allowed})==len(allowed)
    assert len({r['external_relative_path'] for r in allowed})==len(allowed)
    decisions={}
    with decisions_path.open() as stream:
        for row in csv.DictReader(stream):
            assert row['opaque_id'] not in decisions
            decisions[row['opaque_id']]=row
    index=json.loads((source_root/'image_index.json').read_text())
    assert len(index)==audit['source_images']
    by_path={r['opaque_id']:r for r in index}
    assert len(by_path)==len(index)
    provenance=json.loads((source_root/'source_provenance.json').read_text())
    file_provenance={r['path']:r for r in provenance['files']}
    assert len(file_provenance)==len(provenance['files'])
    cards={split:json.loads((source_root/f'{split}_source.json').read_text()) for split in ('train','test')}
    names={split:class_names(card) for split,card in cards.items()}
    assert names['train']==names['test'],'Train/test source class-index schemas differ'
    for split,card in cards.items():
        repo,revision=provenance['sources'][split]
        assert card['id']==repo and card['sha']==revision,'Source card does not match pinned provenance'
    pairs=mapping(course_root/'classes.csv',names['train'])
    by_source_class={p['source_class_index']:p for p in pairs}
    requests=defaultdict(list);requested_hashes=defaultdict(dict);candidates=[]
    for row in allowed:
        decision=decisions[row['opaque_id']]
        assert decision['decision']=='allow','Allowlist includes an excluded/quarantined source ID'
        assert decision['external_relative_path']==row['external_relative_path']
        assert decision['file_sha256']==row['file_sha256']
        for key in ('component_course_train_count','component_course_val_count','component_course_test_count'):
            assert int(decision[key])==0,'Allowlisted source component overlaps course images'
        image=inside(source_root/'images',row['external_relative_path'])
        assert image.is_file() and sha256(image)==row['file_sha256'],'Allowlisted source image changed after audit'
        item=by_path[row['external_relative_path']]
        parquet=inside(source_root,item['parquet'])
        assert item['parquet'] in file_provenance
        assert isinstance(item['row'],int) and item['row']>=0
        requests[item['parquet']].append(item['row'])
        requested_hashes[item['parquet']][item['row']]=row['file_sha256']
        candidates.append((row,item,image))
    labels={};verified_files={}
    for relative,rows in requests.items():
        path=inside(source_root,relative)
        entry=file_provenance[relative]
        assert sha256(path)==entry['sha256'],'Source annotation shard changed from pinned snapshot'
        split=Path(relative).parts[0]
        assert split in cards and entry['repo']==cards[split]['id'] and entry['revision']==cards[split]['sha']
        labels[relative]=selected_labels(path,rows,requested_hashes[relative])
        verified_files[relative]=entry
    eligible=[];discarded=0
    for row,item,image in candidates:
        label=labels[item['parquet']][item['row']]
        if label not in by_source_class:
            discarded+=1;continue
        pair=by_source_class[label]
        split=Path(item['parquet']).parts[0]
        eligible.append({'external_id':row['opaque_id'],'external_relative_path':row['external_relative_path'],
            'image_path':str(image),'image_sha256':row['file_sha256'],**pair,
            'source_dataset':cards[split]['id'],'source_revision':cards[split]['sha'],
            'source_parquet':item['parquet'],'source_parquet_row':item['row'],
            'source_parquet_sha256':file_provenance[item['parquet']]['sha256']})
    counts=Counter(r['course_label'] for r in eligible)
    output=Path(output_root)/uuid.uuid4().hex[:12];output.mkdir(parents=True,exist_ok=False)
    manifest={'version':VERSION,'training_images':eligible,'source_image_root':str((source_root/'images').resolve()),
              'source_provenance_sha256':sha256(source_root/'source_provenance.json'),
              'image_index_sha256':sha256(source_root/'image_index.json'),
              'audit_allowlist_sha256':sha256(allow_path),'audit_summary_sha256':sha256(summary_path),
              'audit_decisions_sha256':sha256(decisions_path),'course_classes_sha256':sha256(course_root/'classes.csv'),
              'source_cards_sha256':{split:sha256(source_root/f'{split}_source.json') for split in cards},
              'source_shards_verified':verified_files,'class_mapping':pairs,
              'summary':{'audit_allowed_images':len(allowed),'source_label_rows_interpreted':len(candidates),
                  'audit_protected_direct_coverage':audit.get('protected_direct_coverage',{}),
                  'eligible_target_class_images':len(eligible),'discarded_other_class_images':discarded,
                  'per_course_class':{str(p['course_label']):counts[p['course_label']] for p in pairs},
                  'excluded_source_labels_interpreted':False,'protected_test_labels_joined_or_exported':False,
                  'course_files_modified':False,'trained':False,
                  'limitations':'Image-only overlap audit is heuristic, not proof of absent transformed duplicates. Only completed-audit allowlisted IDs were labeled; no source labels were joined to course held-out IDs.'}}
    (output/'external_training_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'summary.json').write_text(json.dumps(manifest['summary'],indent=2))
    with (output/'class_mapping.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    print('MANIFEST_COMPLETE',output.resolve(),json.dumps(manifest['summary']),flush=True)
    return output,manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True);parser.add_argument('--audit-dir',required=True)
    parser.add_argument('--course-root',required=True);parser.add_argument('--output-root',default='outputs/Cars196_manifest')
    args=parser.parse_args(argv);build(args.source_root,args.audit_dir,args.course_root,args.output_root)

if __name__=='__main__':main()
