#!/usr/bin/env python3
"""Complete local-video CPU oracle comparison; never a hardware/FPS acceptance."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    gen=importlib.import_module(name+'.gen2')
    geometry=importlib.import_module(name+'.geometry')
    geometry.FrameGeometry.check=lambda self:None
    clock=[0.]
    gen.time=SimpleNamespace(monotonic=lambda:clock[0])
    return gen,clock


def expected_confirmations(trial, version):
    matches = [v for v in trial['versions'] if v['version'] == version]
    if len(matches) != 1:
        raise ValueError('Reference needs exactly one result for '+version)
    return matches[0]['confirmed_frames']


def check(baseline,package,reference,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    modules=[load(Path(baseline)/'dohc/infrared','frozen_ir'),load(Path(package)/'_core','packaged_ir')]
    version=modules[0][0].module_identity()['version']
    assert modules[1][0].module_identity()['version']==version
    reference=json.loads(Path(reference).read_text())
    report=dict(mode='CPU_oracle_unbudgeted_no_recording_inventory',hardware_test=False,
        absolute_accuracy=None,algorithm_version=version,upstream=str(baseline),package=str(package),trials=[])
    started=time.monotonic()
    for trial in reference['trials']:
        evidence=Path(trial['evidence'])
        sources=trial['sources']
        for path,key in [(sources['video'],'video_sha256'),
            (evidence/'device/calibration.json','calibration_sha256'),
            (evidence/'device/patterns.json','patterns_sha256')]:
            assert sha(path)==sources[key]
        session=Path(sources['video']).parents[1]
        times_path=session/'times/cam0.json'
        assert sha(times_path)==sources['timestamps_sha256']
        timing=json.loads(times_path.read_text())
        assert timing['unit']=='ns' and timing['clock']=='CLOCK_BOOTTIME'
        stamps=np.asarray(timing['timestamps_ns'],dtype=float)/1e9
        assert np.isfinite(stamps).all() and np.all(np.diff(stamps)>0)
        cal=json.loads((evidence/'device/calibration.json').read_text())
        patterns=json.loads((evidence/'device/patterns.json').read_text())
        trackers=[gen.Gen2Tracker(cal,patterns) for gen,clock in modules]
        capture=cv2.VideoCapture(sources['video'])
        count=confirmed=0;maxima=dict(points_px=0.,rvec_rad=0.,translation_m=0.,residual_px=0.)
        differences=[]
        with (output/(evidence.name+'.csv')).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['frame','confirmed','mode','differences'])
            try:
                while True:
                    ok,image=capture.read()
                    if not ok:break
                    results=[None,None]
                    for j in ([0,1] if count%2 else [1,0]):
                        modules[j][1][0]=float(stamps[count])
                        results[j]=trackers[j].detect(image,float(stamps[count]))
                    a,b=results
                    errors=[]
                    for key in ('search_mode','quality','ambiguous_pattern_ids'):
                        if a.get(key)!=b.get(key):errors.append(key)
                    for key in ('roi_prediction','roi_coverage','recovery_retry'):
                        if a.get('search',{}).get(key)!=b.get('search',{}).get(key):
                            errors.append('search.'+key)
                    da={d['pattern_id']:d for d in a['detections']}
                    db={d['pattern_id']:d for d in b['detections']}
                    if da.keys()!=db.keys():errors.append('identities')
                    for name in da.keys()&db.keys():
                        x,y=da[name],db[name]
                        for key in ('led_ids','quality','pose_ambiguous','threshold','correspondence_search','pose_resolution'):
                            if x.get(key)!=y.get(key):errors.append(name+'.'+key)
                        for field,label,tolerance in [('points_raw','points_px',.001)]:
                            delta=float(np.max(np.abs(np.array(x[field])-y[field])))
                            maxima[label]=max(maxima[label],delta)
                            if delta>tolerance:errors.append(name+'.'+field)
                        for field,label,tolerance in [('rvec','rvec_rad',.001),('tvec_m','translation_m',1e-5),
                            ('rms_px','residual_px',.001),('max_error_px','residual_px',.001)]:
                            delta=float(np.max(np.abs(np.array(x['pose'][field])-y['pose'][field])))
                            maxima[label]=max(maxima[label],delta)
                            if delta>tolerance:errors.append(name+'.pose.'+field)
                    if errors:differences.append(dict(frame=count,fields=errors))
                    confirmed+=int(bool(db))
                    writer.writerow([count,bool(db),b['search_mode'],';'.join(errors)])
                    count+=1
                    if count%600==0:print(evidence.name,count,flush=True)
            finally:capture.release()
        assert count==len(stamps)==trial['frames']
        expected=expected_confirmations(trial,version)
        result=dict(name=evidence.name,frames=count,confirmed=confirmed,expected=expected,
            maxima=maxima,differences=differences,input_hashes=sources,
            passed=not differences and confirmed==expected)
        report['trials'].append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'}),flush=True)
    report['wall_seconds']=time.monotonic()-started
    report['passed']=all(t['passed'] for t in report['trials'])
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','package','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();check(a.baseline,a.package,a.reference,a.output)
