"""Consume the existing capture source; do not open a second camera."""
from dohc_ir_tracker import Tracker

def run(source, config='config/config.example.json'):
    """source yields (native_jpeg_bytes, sequence, arrival_monotonic)."""
    last_sequence = None
    with Tracker(config=config) as tracker:
        for jpeg_bytes, sequence, arrival_monotonic in source:
            tracker.submit(jpeg_bytes, sequence, arrival_monotonic)
            result = tracker.latest_result()
            if result.get('source_sequence') == last_sequence:
                continue
            last_sequence = result.get('source_sequence')
            for target in result.get('detections', []):
                if target['quality']['pose_usable']:
                    print(target['pattern_id'], target['pose'],
                          'orientation_supported=', target['quality']['orientation_supported'])
