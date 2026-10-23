import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sdk/src'))
sys.path.insert(0,str(ROOT/'dohc'))
from dohc_ir_tracker import Tracker
from dohc_ir_tracker import tracker as sdk
from dohc_ir_tracker.cli import validate_manifest, doctor
from dohc_ir_tracker.report import generate


@pytest.fixture
def config(tmp_path):
    p=tmp_path/'config.json'
    p.write_text(json.dumps(dict(calibration=str(ROOT/'sdk/config/calibration-v1-940nm.json'),
        patterns=str(ROOT/'sdk/config/patterns-p01-p05.json'),optical_profile_confirmed=True)))
    return p


class Worker:
    timeout=.1
    def __init__(self,*args):
        self.state=dict(state='ready',fresh=False,detections=[],observations=[])
        self.calls=[]
        self.closed=False
    def snapshot(self):return self.state.copy()
    def set_recording_session(self,started):self.started=started
    def submit(self,frame,sequence,**meta):
        self.calls.append((frame,sequence,meta));return True
    def metrics_for_recording(self,started):return {'completed':len(self.calls)}
    def close(self):self.closed=True


def start(config):
    with patch.object(sdk,'require_platform'),patch.object(sdk,'_create_worker',side_effect=Worker):
        return Tracker(config).start()


def test_auto_metadata_and_immutable_producer_buffer(config):
    tracker=start(config)
    frame=bytearray(b'\xff\xd8jpeg')
    assert tracker.submit(frame)
    frame[-1]=0
    packet=tracker._worker.calls[0]
    assert packet[0]==b'\xff\xd8jpeg'
    assert packet[1]==0 and packet[2]['source_monotonic']>=tracker._start
    assert packet[2]['source_time_kind']=='sdk_frame_receipt'
    assert tracker.submit(b'\xff\xd8new',7,time.monotonic())
    assert not tracker.submit(b'\xff\xd8duplicate',7)
    assert not tracker.submit(b'\xff\xd8older',6)
    assert tracker.submit(b'\xff\xd8auto')
    assert tracker._worker.calls[-1][1]==8
    tracker.close()


def test_lifecycle_and_snapshot_copies(config):
    tracker=start(config)
    worker=tracker._worker
    worker.state['detections']=[{'pose':{'tvec_m':[1,2,3]}}]
    snapshot=tracker.latest_result()
    snapshot['detections'][0]['pose']['tvec_m'][0]=999
    assert tracker.latest_result()['detections'][0]['pose']['tvec_m'][0]==1
    with pytest.raises(RuntimeError):tracker.start()
    tracker.stop()
    assert worker.closed and not tracker.latest_result()['detections']
    with patch.object(sdk,'require_platform'),patch.object(sdk,'_create_worker',side_effect=Worker):
        tracker.start()
    assert tracker._sequence==-1 and tracker._worker is not worker
    tracker.close();tracker.close()
    with pytest.raises(RuntimeError):tracker.start()


@pytest.mark.parametrize('frame,stamp,error',[(b'bad',None,ValueError),('bad',None,TypeError),
    (b'\xff\xd8x',float('nan'),ValueError),(b'\xff\xd8x',True,ValueError)])
def test_invalid_inputs(config,frame,stamp,error):
    tracker=start(config)
    try:
        with pytest.raises(error):tracker.submit(frame,source_monotonic=stamp)
    finally:tracker.close()


def test_worker_error_exposed_and_closed(config):
    tracker=start(config)
    worker=tracker._worker
    worker.state=dict(state='error',error='MPP test failure')
    with pytest.raises(RuntimeError,match='MPP test failure'):tracker.latest_result()
    tracker.close()
    assert worker.closed


def test_initialization_failure_closes_worker(config):
    worker=Worker()
    worker.state=dict(state='error',error='Mali test failure')
    with patch.object(sdk,'require_platform'),patch.object(sdk,'_create_worker',return_value=worker):
        with pytest.raises(RuntimeError,match='Mali'):Tracker(config).start()
    assert worker.closed


def test_session_manifest_and_metrics_cleanup(config,tmp_path):
    value=json.loads(config.read_text());value['output_dir']=str(tmp_path/'runs')
    config.write_text(json.dumps(value))
    identity=SimpleNamespace(module_identity=lambda:{'version':'2.0.0-dev.19.4'})
    with patch.dict(sys.modules, {'dohc_ir_tracker._core.version':identity}):
        tracker=start(config)
    manifest=json.loads((tracker.output_directory/'session.json').read_text())
    assert len(manifest['input_files']['calibration'])==64
    worker=tracker._worker
    def failed_metrics(_):raise OSError('metrics test')
    worker.metrics_for_recording=failed_metrics
    with pytest.raises(OSError,match='metrics test'):tracker.stop()
    assert worker.closed and not tracker.latest_result()['fresh']
    tracker.close()


@pytest.mark.parametrize('tag,python', [('cp39-cp39-linux_aarch64','>=3.8'),
    ('py3-none-any','>=3.9'), ('cp38-cp38-linux_x86_64','>=3.8')])
def test_dependency_lock_rejects_wrong_target(tmp_path,tag,python):
    sys.path.insert(0,str(ROOT/'sdk'))
    from wheel_tools import write_wheel,lock_wheels
    write_wheel(tmp_path,'example','1.0',tag,{},requires_python=python)
    with pytest.raises(ValueError):lock_wheels(tmp_path)


def test_config_requires_optics_confirmation(config):
    x=json.loads(config.read_text());x['optical_profile_confirmed']=False
    config.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='Confirm'):Tracker(config)


def test_no_hardware_fallback_on_host():
    result=doctor()
    assert not result['ok'] and result['hardware_execution']=='not_attempted'


def test_reference_selected_by_algorithm_version():
    sys.path.insert(0,str(ROOT/'sdk'))
    from check_equivalence import expected_confirmations
    trial=dict(versions=[dict(version='2.0.0-dev.19.4',confirmed_frames=1850),
                         dict(version='2.0.0-dev.19.2',confirmed_frames=1844)])
    assert expected_confirmations(trial,'2.0.0-dev.19.4')==1850
    with pytest.raises(ValueError):expected_confirmations(trial,'missing')


def test_toolkit_versions_are_consistent():
    sys.path.insert(0,str(ROOT/'sdk'))
    import build_toolkit
    import dohc_ir_tracker
    from infrared.version import ALGORITHM
    assert build_toolkit.VERSION==dohc_ir_tracker.__version__=='0.0.2'
    assert ALGORITHM['version']=='2.0.0-dev.19.4'
    assert build_toolkit.UPSTREAM=='75b097ee1fcf2371f154d5f30f5609704462e6e4'


def test_manifest_hash_and_timestamp_validation(tmp_path):
    import hashlib
    frame=tmp_path/'frame.jpg';frame.write_bytes(b'\xff\xd8frame')
    p=tmp_path/'manifest.json'
    entries=[dict(file='frame.jpg',source_sequence=1,relative_seconds=0.,sha256=hashlib.sha256(frame.read_bytes()).hexdigest())]
    p.write_text(json.dumps(dict(entries=entries)))
    assert validate_manifest(p)['entries']==entries
    frame.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):validate_manifest(p)
    entries[0]['relative_seconds']=float('nan')
    p.write_text(json.dumps(dict(entries=entries)))
    with pytest.raises(ValueError,match='increasing'):validate_manifest(p)


def test_report_separates_current_pose_and_direction(tmp_path):
    p=tmp_path/'results.jsonl'
    row=dict(state='ready',run_id='run',source_sequence=1,source_monotonic=10.1,
        recording_started_monotonic=10.,completed_monotonic=10.2,fresh=True,search_mode='roi',
        processing_ms=12.,detections=[dict(pattern_id='P03',quality=dict(pose_usable=True,orientation_supported=False))])
    p.write_text(json.dumps(row)+'\n')
    summary=generate(p,tmp_path/'report')
    assert summary['pose_frames']==1 and summary['orientation_supported_frames']==0
    assert summary['absolute_accuracy'] is None
    assert (tmp_path/'report/index.html').is_file()
    with pytest.raises(FileExistsError):generate(p,tmp_path/'report')
    p.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,run_id='other'))+'\n')
    with pytest.raises(ValueError,match='one report per run'):generate(p,tmp_path/'mixed')


def test_real_cpu_worker_packaging_seam(config,tmp_path):
    """CPU is a test oracle only; the public SDK has no CPU selection."""
    import cv2
    import numpy as np
    from infrared.runtime import InfraredWorker
    def factory(cfg,directory):
        return InfraredWorker(sys.executable,cfg['calibration'],cfg['patterns'],backend='cpu',
                              method='gen2',jpeg_decoder='cpu',output=None,profiling=False)
    with patch.object(sdk,'require_platform'),patch.object(sdk,'_create_worker',side_effect=factory):
        tracker=Tracker(config).start()
    try:
        ok,data=cv2.imencode('.jpg',np.zeros((2160,3840,3),np.uint8));assert ok
        tracker.submit(data.tobytes(),1)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            result=tracker.latest_result()
            if result.get('source_sequence')==1:break
            time.sleep(.01)
        assert result.get('source_sequence')==1
        assert result['algorithm']['version']=='2.0.0-dev.19.4'
        assert not (tmp_path/'latest.json').exists()
    finally:tracker.close()
