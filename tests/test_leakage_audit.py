import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
import leakage_audit as audit

class LeakageAuditTests(unittest.TestCase):
    def test_hash_radius_retrieval_and_dual_confirmation(self):
        index=audit.HashIndex(8,10)
        index.add(0,[(0,0)])
        index.add(1,[(0xffff0000ffff0000,0xffff0000ffff0000)])
        matches,_=index.query([((1<<8)-1,(1<<10)-1)])
        self.assertIn(0,matches)
        self.assertEqual(matches[0],(8,10))
        matches,_=index.query([((1<<8)-1,(1<<11)-1)])
        self.assertNotIn(0,matches)
        # Test many arbitrary bit positions, including pigeonhole partition edges.
        rng=np.random.default_rng(5)
        for _ in range(30):
            bits=rng.choice(64,8,replace=False)
            p=sum(1<<int(x) for x in bits)
            self.assertIn(0,index.query([(p,0)])[0])

    def test_transform_calibration_and_decoded_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Training-only synthetic calibration: smooth scene + shapes survives JPEG/resizing.
            from PIL import ImageDraw
            image=Image.new('RGB',(400,300),'lightblue')
            draw=ImageDraw.Draw(image)
            draw.rectangle((45,80,350,240),fill='darkred');draw.ellipse((65,200,125,260),fill='black')
            draw.ellipse((270,200,330,260),fill='black');draw.polygon([(100,80),(150,30),(260,35),(310,80)],fill='navy')
            for i in range(20):draw.line((i*17,0,400-i*11,300),fill=(i*5,60,180),width=1)
            original=root/'original.png';image.save(original)
            metadata=PngImagePlugin.PngInfo();metadata.add_text('description','different metadata')
            rewritten=root/'rewritten.png';image.save(rewritten,pnginfo=metadata)
            first,second=audit.fingerprint(original),audit.fingerprint(rewritten)
            self.assertNotEqual(first['file_sha256'],second['file_sha256'])
            self.assertEqual(first['rgb_sha256'],second['rgb_sha256'])
            index=audit.HashIndex();index.add(0,first['variants'])
            for name,transformed in [('jpeg',image),('resize',image.resize((200,150))),
                ('crop90',image.crop((20,15,380,285))),('crop75',image.crop((50,38,350,263))),
                ('flip',ImageOps.mirror(image))]:
                path=root/(name+'.jpg');transformed.save(path,quality=85)
                result,_=index.query(audit.fingerprint(path)['variants'])
                self.assertIn(0,result,name)

    def test_transitive_quarantine_and_clean_single_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);course=root/'course';external=root/'source'
            for role in ('train','val','test'):
                (course/role).mkdir(parents=True);(course/role/'image.jpg').write_bytes(b'x')
            external.mkdir()
            for name in ('a','b','c','d','e'):(external/f'{name}.jpg').write_bytes(b'x')
            codes={'train':0xAAAAAAAAAAAAAAAA,'val':0,'test':0x5555555555555555,
                   'a':(1<<8)-1,'b':(1<<16)-1,'c':(1<<24)-1,'d':0xf0f0f0f0f0f0f0f0,'e':0xf0f0f0f0f0f0f0f0}
            def fp(path):
                name=path.parent.name if path.parent.name in ('train','val','test') else path.stem
                return {'file_sha256':name,'rgb_sha256':'rgb_'+name,'variants':[(codes[name],codes[name])]*6}
            with patch.object(audit,'fingerprint',side_effect=fp):
                out,summary=audit.audit(external,course,root/'out')
            import pandas as pd
            decisions=pd.read_csv(out/'source_decisions.csv').set_index('external_relative_path')
            for name in ('a.jpg','b.jpg','c.jpg'):
                self.assertEqual(decisions.loc[name,'decision'],'quarantine_heldout_overlap')
                self.assertEqual(decisions.loc[name,'component_course_val_count'],1)
            self.assertEqual(decisions.loc['c.jpg','broad_course_candidate_count'],0)
            self.assertEqual(decisions.loc['d.jpg','decision'],'allow')
            self.assertEqual(decisions.loc['e.jpg','decision'],'exclude_external_duplicate')
            allow=json.loads((out/'allowlist.json').read_text())['images']
            self.assertEqual([r['external_relative_path'] for r in allow],['d.jpg'])
            self.assertFalse(summary['source_labels_read'])
            self.assertFalse(summary['test_labels_recovered_or_exported'])
            self.assertEqual(summary['course_counts'],{'train':1,'val':1,'test':1})
            self.assertNotIn('test_image_id',decisions.columns)

    def test_real_exact_and_rgb_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);course=root/'course';external=root/'source';external.mkdir()
            rng=np.random.default_rng(9)
            images={}
            for role in ('train','val','test'):
                (course/role).mkdir(parents=True)
                image=Image.fromarray(rng.integers(0,256,(70,85,3),dtype=np.uint8))
                path=course/role/'image.png';image.save(path);images[role]=image
            shutil.copyfile(course/'test'/'image.png',external/'exact.png')
            meta=PngImagePlugin.PngInfo();meta.add_text('note','different')
            images['val'].save(external/'rgb.png',pnginfo=meta)
            out,summary=audit.audit(external,course,root/'out')
            self.assertEqual(summary['protected_direct_coverage']['file_exact']['test'],1)
            self.assertEqual(summary['protected_direct_coverage']['rgb_exact']['val'],1)
            self.assertEqual(summary['allowed_source_images'],0)
            self.assertEqual(len(json.loads((out/'allowlist.json').read_text())['images']),0)

if __name__=='__main__':unittest.main()
