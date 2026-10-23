"""Opt-in relocated-wheel smoke test; CPU oracle, never hardware acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest


@pytest.mark.skipif(not os.environ.get('DOHC_TOOLKIT_TEST_BUNDLE'), reason='Requires a relocated built wheel')
def test_relocated_worker_logs_report_and_integrity(tmp_path):
    from dohc_ir_tracker import Tracker, tracker as sdk
    from dohc_ir_tracker._core.runtime import InfraredWorker
    from dohc_ir_tracker._core.version import module_identity
    from dohc_ir_tracker.report import generate
    import cv2
    bundle=Path(os.environ['DOHC_TOOLKIT_TEST_BUNDLE'])
    value=json.loads((bundle/'config/config.example.json').read_text())
    value.update(calibration=str(bundle/'config/calibration-v1-940nm.json'),
        patterns=str(bundle/'config/patterns-p01-p05.json'), optical_profile_confirmed=True,
        diagnostics=True,output_dir=str(tmp_path/'runs'))
    config=tmp_path/'config.json';config.write_text(json.dumps(value))
    def factory(cfg,directory):
        return InfraredWorker(sys.executable,cfg['calibration'],cfg['patterns'],backend='cpu',
            method='gen2',jpeg_decoder='cpu',output=directory/'latest.json',
            log_output=directory/'results.jsonl',profiling=True)
    with patch.object(sdk,'require_platform'),patch.object(sdk,'_create_worker',side_effect=factory):
        tracker=Tracker(config).start()
    try:
        frame=(bundle/'sample/frame.jpg').read_bytes()
        tracker.submit(frame,0)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            result=tracker.latest_result()
            if result.get('source_sequence')==0:break
            time.sleep(.01)
        assert result.get('source_sequence')==0
    finally:tracker.close()
    directory=tracker.output_directory
    assert (directory/'session-summary.json').exists()
    identity=json.loads((directory/'session.json').read_text())['algorithm']
    assert identity['package_version']=='0.0.2'
    assert identity['version']=='2.0.0-dev.19.4'
    assert identity['upstream_commit']=='75b097ee1fcf2371f154d5f30f5609704462e6e4'
    entries=[dict(file='frame.jpg',source_sequence=0,relative_seconds=0.,
                  sha256=hashlib.sha256(frame).hexdigest())]
    (tmp_path/'frame.jpg').write_bytes(frame)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(entries=entries,input_kind='single JPEG packaging smoke fixture')))
    summary=generate(directory/'results.jsonl',tmp_path/'report',manifest,video=True)
    assert summary['completed']==1
    assert '<title>dev.19.4 识别报告</title>' in (tmp_path/'report/index.html').read_text()
    cap=cv2.VideoCapture(str(tmp_path/'report/annotated.mp4'))
    assert cap.read()[0] and not cap.read()[0]
    cap.release()
    module_identity()
    core=Path(sdk.__file__).parent/'_core/config.py'
    original=core.read_bytes()
    try:
        core.write_bytes(original+b'\n# integrity test\n')
        with pytest.raises(ValueError,match='manifest'):module_identity()
    finally:core.write_bytes(original)
