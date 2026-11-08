"""Label-blind, conservative Cars196 overlap audit; never trains or reads annotations.

Near-duplicate detection is heuristic, not proof of absence. Every ambiguous
match is excluded rather than used to transfer source labels to held-out images.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib
import json
import struct
import time
import uuid
import platform
import PIL
import scipy

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
PHASH_RADIUS = 8
DHASH_RADIUS = 10
VARIANT_NAMES = ('full','full_flip','center90','center90_flip','center75','center75_flip')
VERSION = 'Cars196-label-blind-audit-v1'


def image_paths(root):
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(root)
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def bits_to_int(bits):
    value=0
    for bit in np.asarray(bits).reshape(-1): value=(value<<1)|int(bit)
    return value


def perceptual_hashes(image):
    gray=image.convert('L')
    small=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    frequencies=dctn(small,type=2,norm='ortho')[:8,:8]
    # Exclude DC from the threshold; it contains overall brightness.
    median=np.median(frequencies.reshape(-1)[1:])
    phash=bits_to_int(frequencies>median)
    reduced=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16)
    dhash=bits_to_int(reduced[:,1:]>reduced[:,:-1])
    return phash,dhash


def fingerprint(path):
    file_hash=file_sha256(path)
    with Image.open(path) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
        rgb=hashlib.sha256(struct.pack('>II',*image.size)+image.tobytes()).hexdigest()
        variants=[]
        width,height=image.size
        for fraction in (1.0,0.9,0.75):
            w,h=max(1,round(width*fraction)),max(1,round(height*fraction))
            left,top=(width-w)//2,(height-h)//2
            crop=image.crop((left,top,left+w,top+h))
            variants.extend((perceptual_hashes(crop),perceptual_hashes(ImageOps.mirror(crop))))
    return {'file_sha256':file_hash,'rgb_sha256':rgb,'variants':variants}


class Components:
    def __init__(self,n): self.parent=list(range(n));self.size=[1]*n
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b]


class HashIndex:
    """Exact-radius pHash candidate retrieval, followed by dHash confirmation.

    With radius r and r+1 disjoint chunks, any pair at distance <=r must
    share at least one exact chunk. This avoids approximate LSH omissions.
    """
    def __init__(self,phash_radius=PHASH_RADIUS,dhash_radius=DHASH_RADIUS):
        assert 0<=phash_radius<64 and 0<=dhash_radius<=64
        self.pr,self.dr=phash_radius,dhash_radius
        self.parts=[]
        offset=0
        for i in range(phash_radius+1):
            width=64//(phash_radius+1)+(i<64%(phash_radius+1))
            self.parts.append((offset,(1<<width)-1));offset+=width
        self.buckets=defaultdict(list)
        self.phash=[];self.dhash=[];self.owners=[]
    def add(self,owner,variants):
        for p,d in set(tuple(v) for v in variants):
            row=len(self.phash);self.phash.append(p);self.dhash.append(d);self.owners.append(owner)
            for part,(offset,mask) in enumerate(self.parts):self.buckets[(part,(p>>offset)&mask)].append(row)
    def query(self,variants):
        matched={};nearest=None
        for p,d in variants:
            lists=[self.buckets.get((part,(p>>offset)&mask),[]) for part,(offset,mask) in enumerate(self.parts)]
            candidates=np.unique(np.fromiter((row for values in lists for row in values),dtype=np.int64))
            if not len(candidates):continue
            # bitwise_count is available in NumPy>=2; fall back without relaxing checks.
            pp=np.fromiter((self.phash[i] for i in candidates),dtype=np.uint64,count=len(candidates))
            dd=np.fromiter((self.dhash[i] for i in candidates),dtype=np.uint64,count=len(candidates))
            count=getattr(np,'bitwise_count',None)
            if count is None:
                pd=np.fromiter((int(x).bit_count() for x in (pp^np.uint64(p))),dtype=np.int16)
                dh=np.fromiter((int(x).bit_count() for x in (dd^np.uint64(d))),dtype=np.int16)
            else:
                pd=count(pp^np.uint64(p));dh=count(dd^np.uint64(d))
            closest=int(np.argmin(pd.astype(np.int16)*65+dh))
            pair=(int(pd[closest]),int(dh[closest]))
            if nearest is None or pair<nearest:nearest=pair
            for row,distp,distd in zip(candidates[(pd<=self.pr)&(dh<=self.dr)],pd[(pd<=self.pr)&(dh<=self.dr)],dh[(pd<=self.pr)&(dh<=self.dr)]):
                owner=self.owners[int(row)];pair=(int(distp),int(distd))
                if owner not in matched or pair<matched[owner]:matched[owner]=pair
        return matched,nearest


def audit(external_root,course_root,output_root,phash_radius=PHASH_RADIUS,dhash_radius=DHASH_RADIUS):
    external_root,course_root=Path(external_root),Path(course_root)
    externals=image_paths(external_root)
    if not externals:raise ValueError('No external images found')
    # Course roles come only from directories; source class annotations are never opened.
    protected=[]
    for role in ('train','val','test'):
        paths=image_paths(course_root/role)
        if not paths:raise ValueError(f'No course {role} images')
        protected.extend((role,p) for p in paths)
    rows=[{'role':role,'path':p,'opaque_id':None} for role,p in protected]
    start_external=len(rows)
    rows += [{'role':'external','path':p,'opaque_id':f'source_{i:06d}'} for i,p in enumerate(externals)]
    components=Components(len(rows));index=HashIndex(phash_radius,dhash_radius)
    exact_file={};exact_rgb={};coverage={kind:defaultdict(set) for kind in ('file_exact','rgb_exact','near_direct','broad_ambiguous')}
    diagnostics={};protected_fingerprints=[];protected_arrays=None;start=time.perf_counter()
    for i,row in enumerate(rows):
        fp=fingerprint(row['path'])
        exact_matches=set()
        for field,lookup,kind in [('file_sha256',exact_file,'file_exact'),('rgb_sha256',exact_rgb,'rgb_exact')]:
            previous=lookup.get(fp[field],[])
            for j in previous:
                components.union(i,j);exact_matches.add(j)
                if row['role']=='external' and rows[j]['role']!='external':coverage[kind][rows[j]['role']].add(j)
            lookup.setdefault(fp[field],[]).append(i)
        near,nearest=index.query(fp['variants'])
        for j in near:
            components.union(i,j)
            if row['role']=='external' and rows[j]['role']!='external':coverage['near_direct'][rows[j]['role']].add(j)
        broad=set()
        if row['role']!='external':
            protected_fingerprints.append(fp['variants'])
        else:
            if protected_arrays is None:
                ph,dh,owners=[],[],[]
                for owner,variants in enumerate(protected_fingerprints):
                    for p,d in variants:ph.append(p);dh.append(d);owners.append(owner)
                protected_arrays=(np.asarray(ph,dtype=np.uint64),np.asarray(dh,dtype=np.uint64),np.asarray(owners))
            ph,dh,owners=protected_arrays
            count=getattr(np,'bitwise_count',None)
            for p,d in fp['variants']:
                pd=(count(ph^np.uint64(p)) if count is not None else np.fromiter((int(x).bit_count() for x in ph^np.uint64(p)),dtype=np.int16))
                dd=(count(dh^np.uint64(d)) if count is not None else np.fromiter((int(x).bit_count() for x in dh^np.uint64(d)),dtype=np.int16))
                broad.update(int(x) for x in owners[(pd<=12)|(dd<=10)])
            # Broad source-course ambiguity is quarantined; it never licenses label recovery.
            for j in broad:
                components.union(i,j);coverage['broad_ambiguous'][rows[j]['role']].add(j)
        index.add(i,fp['variants'])
        if row['role']=='external':
            diagnostics[i]={'opaque_id':row['opaque_id'],'external_relative_path':str(row['path'].relative_to(external_root)),
                'file_sha256':fp['file_sha256'],'rgb_sha256':fp['rgb_sha256'],
                'direct_match_count':len(set(near)|exact_matches),'broad_course_candidate_count':len(broad),
                'nearest_index_candidate_phash':None if nearest is None else nearest[0],
                'dhash_at_nearest_index_candidate':None if nearest is None else nearest[1]}
        if (i+1)%200==0:print('AUDIT',i+1,'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
    members=defaultdict(list)
    for i in range(len(rows)):members[components.find(i)].append(i)
    decisions=[];clean=[];component_summary=[]
    for member_indices in members.values():
        role_counts=Counter(rows[i]['role'] for i in member_indices if rows[i]['role']!='external')
        external_indices=sorted(i for i in member_indices if rows[i]['role']=='external')
        if not external_indices:continue
        component_id='component_'+hashlib.sha256('|'.join(rows[i]['opaque_id'] for i in external_indices).encode()).hexdigest()[:16]
        if role_counts['val'] or role_counts['test']: reason='quarantine_heldout_overlap'
        elif role_counts['train']:reason='exclude_training_duplicate'
        else:reason='clean_component'
        component_summary.append({'component_id':component_id,'external_count':len(external_indices),
            'course_train_count':role_counts['train'],'course_val_count':role_counts['val'],
            'course_test_count':role_counts['test'],'reason':reason})
        for position,i in enumerate(external_indices):
            status='allow' if reason=='clean_component' and position==0 else (
                'exclude_external_duplicate' if reason=='clean_component' else reason)
            decision=diagnostics[i]|{'component_id':component_id,'decision':status,
                'component_course_train_count':role_counts['train'],'component_course_val_count':role_counts['val'],
                'component_course_test_count':role_counts['test']}
            decisions.append(decision)
            if status=='allow':clean.append({'opaque_id':rows[i]['opaque_id'],'external_relative_path':str(rows[i]['path'].relative_to(external_root)),
                                            'file_sha256':diagnostics[i]['file_sha256']})
    output=Path(output_root)/uuid.uuid4().hex[:12]
    output.mkdir(parents=True,exist_ok=False)
    import pandas as pd
    pd.DataFrame(sorted(decisions,key=lambda x:x['opaque_id'])).to_csv(output/'source_decisions.csv',index=False)
    pd.DataFrame(component_summary).to_csv(output/'duplicate_components.csv',index=False)
    (output/'allowlist.json').write_text(json.dumps({'audit_version':VERSION,'external_root':str(external_root.resolve()),'images':clean},indent=2))
    summary={'version':VERSION,'audit_script_sha256':file_sha256(Path(__file__)),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'Pillow':PIL.__version__,'scipy':scipy.__version__},
        'source_images':len(externals),'course_counts':dict(Counter(role for role,_ in protected)),
        'decision_counts':dict(Counter(r['decision'] for r in decisions)),'allowed_source_images':len(clean),
        'protected_direct_coverage':{kind:{role:len(values.get(role,set())) for role in ('train','val','test')}
                                     for kind,values in coverage.items()},
        'phash_radius':phash_radius,'dhash_radius':dhash_radius,'near_match_rule':'pHash<=radius AND dHash<=radius for any variant pair',
        'broad_source_course_rule':'pHash<=12 OR dHash<=10; ambiguous candidates excluded','variant_names':VARIANT_NAMES,'connected_component_exclusion':True,'source_labels_read':False,
        'test_labels_recovered_or_exported':False,'trained':False,'elapsed_seconds':time.perf_counter()-start,
        'limitations':'Heuristic audit, not proof of absent overlap. Unrepresented crops/edits can evade hashes; ambiguous candidates are excluded. No source annotations were read. Direct coverage is reported by role only, never with source-label/test-ID joins.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('AUDIT_COMPLETE',output.resolve(),json.dumps(summary),flush=True)
    return output,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-image-root',required=True)
    parser.add_argument('--course-root',required=True)
    parser.add_argument('--output-root',default='outputs/Cars196_audit')
    args=parser.parse_args(argv)
    audit(args.external_image_root,args.course_root,args.output_root)

if __name__=='__main__':main()
