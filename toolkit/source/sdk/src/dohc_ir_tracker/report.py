"""File-only reports; never manufacture a pose for an unprocessed frame."""
import csv
import html
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def read_results(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError('Invalid result JSONL line %d' % number) from exc
        rows.append(value)
    if len({r.get('run_id') for r in rows if r.get('source_sequence') is not None}) > 1:
        raise ValueError('Generate one report per run; separate session clocks cannot share an FPS window')
    seen = set()
    for row in rows:
        key = (row.get('run_id'), row.get('source_sequence'))
        if row.get('state') == 'ready' and key[1] is not None:
            if key in seen:
                raise ValueError('Duplicate completed source frame: %s' % (key,))
            seen.add(key)
    return rows


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*p))]


def summarize(rows):
    completed = [r for r in rows if r.get('state') == 'ready' and r.get('source_sequence') is not None]
    fresh = [r for r in completed if r.get('fresh')]
    detected = [r for r in fresh if r.get('detections')]
    times = [r['completed_monotonic'] for r in completed if 'completed_monotonic' in r]
    starts = [r['recording_started_monotonic'] for r in completed
              if r.get('recording_started_monotonic') is not None]
    origin = min(starts) if starts else min(times) if times else None
    duration = max(times)-origin if times else None
    pose = [r for r in fresh if any(d.get('quality', {}).get('pose_usable') for d in r.get('detections', []))]
    supported = [r for r in fresh if any(d.get('quality', {}).get('orientation_supported') for d in r.get('detections', []))]
    gaps = sorted(r['completed_monotonic'] for r in detected if 'completed_monotonic' in r)
    gap = max(b-a for a, b in zip([origin]+gaps, gaps+[max(times)])) if times else None
    stages = {}
    for mode in sorted({r.get('search_mode', 'unknown') for r in completed}):
        group = [r for r in completed if r.get('search_mode', 'unknown') == mode]
        stages[mode] = {}
        for name in ('decode_ms', 'processing_ms', 'rga_ms', 'ipc_roundtrip_ms', 'publish_ms'):
            values = [r[name] for r in group if isinstance(r.get(name), (float, int))]
            stages[mode][name] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                p50=percentile(values, .5), p95=percentile(values, .95), maximum=max(values) if values else None)
    hz = lambda n: n/duration if duration and duration > 0 else None
    return dict(completed=len(completed), fresh_confirmed=len(detected), pose_frames=len(pose),
        orientation_supported_frames=len(supported), observed_duration_seconds=duration,
        processed_fps=hz(len(completed)), pose_hz=hz(len(pose)), orientation_supported_hz=hz(len(supported)),
        longest_confirmed_gap_seconds=gap, timing_by_mode=stages, absolute_accuracy=None,
        window='session start through last recorded completion; not a fixed-duration device acceptance',
        errors=[r for r in rows if r.get('state') == 'error'])


def annotate(manifest_path, rows, output):
    import cv2
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['entries']
    if not entries:
        raise ValueError('Empty input manifest')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    run_ids = {r.get('run_id') for r in rows if r.get('source_sequence') is not None}
    if len(run_ids) > 1:
        raise ValueError('Annotate one run at a time; source sequences can repeat across sessions')
    lookup = {r['source_sequence']: r for r in rows if r.get('source_sequence') is not None}
    from .cli import validate_manifest
    validate_manifest(manifest_path)
    first = cv2.imread(str(manifest_path.parent/entries[0]['file']))
    if first is None:
        raise ValueError('Cannot decode report input')
    height, width = first.shape[:2]
    width_out, height_out = 1280, round(height*1280/width)//2*2
    offsets = [e['relative_seconds'] for e in entries]
    fps = (len(offsets)-1)/(offsets[-1]-offsets[0]) if len(offsets) > 1 else 1.
    process = subprocess.Popen([ffmpeg, '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', '%dx%d' % (width_out, height_out), '-framerate', str(fps), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    try:
        for entry in entries:
            image = cv2.imread(str(manifest_path.parent/entry['file']))
            if image is None or image.shape != first.shape:
                raise ValueError('Missing or inconsistent report frame')
            row = lookup.get(entry['source_sequence'])
            detections = row.get('detections', []) if row and row.get('fresh') else []
            for detection in detections:
                color = (0, 220, 0) if detection.get('quality', {}).get('orientation_supported') else (0, 165, 255)
                for point in detection.get('points_raw', []):
                    cv2.circle(image, tuple(round(v) for v in point), 10, color, 2)
                pose = detection.get('pose')
                if pose:
                    import numpy as np
                    cal = manifest.get('calibration')
                    if cal:
                        axes = np.array([[0,0,0],[.03,0,0],[0,.03,0],[0,0,.03]], dtype=float)
                        pixels = cv2.fisheye.projectPoints(axes.reshape(1,-1,3),
                            np.array(pose['rvec']), np.array(pose['tvec_m']),
                            np.array(cal['K']), np.array(cal['D']))[0].reshape(-1,2)
                        if np.isfinite(pixels).all():
                            origin = tuple(np.rint(pixels[0]).astype(int))
                            for p in pixels[1:]:
                                cv2.line(image, origin, tuple(np.rint(p).astype(int)), color, 3)
            image = cv2.resize(image, (width_out, height_out))
            status = 'CURRENT SOURCE POSE' if detections else 'PROCESSED / NO FRESH DETECTION' if row else 'NO LOGGED RESULT'
            cv2.putText(image, '%d | %.3fs | %s' % (entry['source_sequence'], entry['relative_seconds'], status),
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait(timeout=60):
            raise RuntimeError('H264 encoder failed')
    except BaseException:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise


def generate(results, output, input_manifest=None, video=False):
    rows = read_results(results)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Report output already exists')
    output.mkdir(parents=True)
    summary = summarize(rows)
    (output/'statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    with (output/'frames.csv').open('w', newline='') as stream:
        fields = ['source_sequence', 'source_monotonic', 'state', 'fresh', 'search_mode',
                  'decode_ms', 'processing_ms', 'confirmed_targets', 'orientation_supported']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict({k:r.get(k) for k in fields[:-2]},
                confirmed_targets=';'.join(d['pattern_id'] for d in r.get('detections', [])),
                orientation_supported=any(d.get('quality', {}).get('orientation_supported') for d in r.get('detections', []))))
    representative = next((r for r in rows if r.get('fresh') and r.get('detections')
                           and r.get('search_mode') == 'roi'),
                          next((r for r in rows if r.get('state') == 'ready'), {}))
    (output/'representative-frame.json').write_text(json.dumps(representative, ensure_ascii=False, indent=2)+'\n')
    if video:
        if input_manifest is None:
            raise ValueError('--video requires --input-manifest')
        annotate(input_manifest, rows, output/'annotated.mp4')
    def escaped(value):
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))
    movie = '<video controls preload="metadata" src="annotated.mp4"></video>' if video else ''
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>dev.19.4 识别报告</title>
<link rel="icon" href="data:,">
<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4f6fa;color:#17233c}
section,details{background:white;padding:20px;margin:18px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
video{width:100%%}summary{cursor:pointer;font-weight:600}a{color:#174ea6}</style>
<h1>dev.19.4 · 红外标靶识别报告</h1><p>只显示同一源帧的观测；方向支持不等于真值认证。绝对精度：N/A。</p>
<nav><a href="#segment">整段统计</a> · <a href="#frame">代表单帧</a> · <a href="frames.csv">逐帧CSV</a> · <a href="statistics.json">统计JSON</a></nav>
<section id="segment"><h2>整段结果及按搜索模式耗时</h2><p>FPS（每秒处理帧数）与位姿Hz（每秒输出次数）分别统计，窗口截止于最后一条完成记录。</p><pre>%s</pre></section>
<section id="frame"><h2>代表单帧</h2><p>JPEG（静态图像压缩格式）→ MPP（媒体处理框架）→ RGA（二维图像加速器）局部转换 → GPU（图形处理器）提点 → IPPE（平面位姿求解）＋G（稳健精修）→时序选解。</p>
<details><summary>展开单帧输入、候选、质量与耗时</summary><pre>%s</pre></details></section>
<details><summary>字段及计时解释</summary><p>ROI（感兴趣区域）；Condition（条件变量）；pending（最新待处理引用）；worker（识别子进程）。processing_ms是识别墙钟，包含RGA；decode_ms是JPEG到YUV（亮度色度格式），不能再次累加父子嵌套耗时。profile（诊断计时）可能未开启，未记录不等于零。</p></details>
<section><h2>回放说明</h2><p>预览按输入平均帧率显示，原始时间以CSV和输入清单为准；画面显示源帧结果，不能当成发布时刻。MP4恢复输入经过有损编码，不能用于原始像素精度验收。橙色轴表示方向仍未获支持。</p>%s</section></html>''' % (escaped(summary), escaped(representative), movie)
    (output/'index.html').write_text(text)
    return summary
