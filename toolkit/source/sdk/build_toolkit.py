#!/usr/bin/env python3
"""Build the standalone dev.19.4 wheel and an audited offline bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from wheel_tools import write_wheel, lock_wheels

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM='75b097ee1fcf2371f154d5f30f5609704462e6e4'
VERSION='0.0.2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    if (ROOT/'SOURCE.json').exists():
        manifest=json.loads((ROOT/'SOURCE.json').read_text())
        if args == ('status','--porcelain'):
            return 'source snapshot' if manifest['source_dirty'] else ''
        if args == ('rev-parse','HEAD'):
            return manifest['source_commit']
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def build(output,native_root,dependencies,preview=False):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    dirty=bool(git('status','--porcelain'))
    if dirty and not preview:
        raise RuntimeError('Commit the tested source first, or use --preview for an explicitly dirty build')
    stage=output/'stage'
    package=stage/'dohc_ir_tracker'
    shutil.copytree(ROOT/'sdk/src/dohc_ir_tracker',package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    core=package/'_core'
    core.mkdir()
    files=[p for p in (ROOT/'dohc/infrared').iterdir() if p.suffix in ('.py','.csv','.json')]
    protected=['gen2.py','geometry.py','detector.py','config.py','pose_resolution.py','roi_motion.py',
               'roi_association.py','profiling.py','opencl_extract.py','image_source.py','methods.py','led_coordinates.csv']
    upstream_hashes={}
    for name in protected:
        if (ROOT/'SOURCE.json').exists():
            upstream_hashes[name]=json.loads((ROOT/'SOURCE.json').read_text())['upstream_frozen_sha256'][name]
        else:
            frozen=subprocess.check_output(['git','-C',str(ROOT),'show',UPSTREAM+':dohc/infrared/'+name])
            upstream_hashes[name]=hashlib.sha256(frozen).hexdigest()
        if sha(ROOT/'dohc/infrared'/name) != upstream_hashes[name]:
            raise ValueError('Frozen algorithm changed: '+name)
    for p in files:
        shutil.copy2(p,core/p.name)
    native_root=Path(native_root)
    for directory,schema in [('infrared_codec','dohc.ir.codec-build.v1'),('infrared_assignment','dohc.ir.assignment-build.v1')]:
        source=ROOT/'native'/directory
        target=package/'native'/directory
        target.mkdir(parents=True)
        for p in source.iterdir():
            if p.suffix in ('.cpp','.h','.txt'):
                shutil.copy2(p,target/p.name)
        binary=native_root/'native'/directory/('libdohc_ir_'+('codec' if directory=='infrared_codec' else 'assignment')+'.so')
        metadata=json.loads(binary.with_name('build-info.json').read_text())
        data=binary.read_bytes()
        sources={p.name:sha(p) for p in target.iterdir()}
        if (len(data)<20 or data[:6]!=b'\x7fELF\x02\x01' or int.from_bytes(data[18:20],'little')!=183 or
                metadata.get('schema')!=schema or metadata.get('abi')!=1 or
                metadata.get('library_sha256')!=sha(binary) or metadata.get('source_sha256')!=sources):
            raise ValueError('Native ARM64 library/source/build manifest mismatch: '+directory)
        shutil.copy2(binary,target/binary.name)
        shutil.copy2(binary.with_name('build-info.json'),target/'build-info.json')
    manifest=dict(schema='dohc.ir.toolkit-build.v1',package_version=VERSION,
        algorithm_version='2.0.0-dev.19.4',upstream_commit=UPSTREAM,source_commit=git('rev-parse','HEAD'),
        source_dirty=dirty,upstream_frozen_sha256=upstream_hashes,
        core_sha256={p.name:sha(p) for p in core.iterdir()},
        target=dict(os='Linux',machine='aarch64',python='CPython 3.8',
                    image='RK3588 Ubuntu 20.04; existing MPP/RGA/Mali stack required'),
        hardware_validation='pending; no device contacted during this packaging task')
    (package/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    wheelhouse=output/'wheelhouse'
    wheelhouse.mkdir()
    dependencies=Path(dependencies)
    for p in dependencies.glob('*.whl'):
        shutil.copy2(p,wheelhouse/p.name)
    wheel_files={p.relative_to(stage).as_posix():p.read_bytes() for p in stage.rglob('*') if p.is_file()}
    wheel=write_wheel(wheelhouse,'dohc-ir-tracker',VERSION,'py3-none-linux_aarch64',wheel_files,
        ['numpy==1.23.5','scipy==1.10.1','opencv-python-headless==4.13.0.92','pyopencl==2019.1.1','imageio-ffmpeg==0.5.1'])
    (output/'requirements.lock').write_text(lock_wheels(wheelhouse))
    for name in ('install.sh','verify_bundle.py','README.md','THIRD_PARTY.md','CHANGELOG.md'):
        shutil.copy2(ROOT/'sdk'/name,output/name)
    shutil.copytree(ROOT/'sdk/config',output/'config')
    shutil.copytree(ROOT/'sdk/examples',output/'examples')
    if (ROOT/'sdk/sample').exists():
        shutil.copytree(ROOT/'sdk/sample',output/'sample')
    shutil.copytree(ROOT/'sdk/tests',output/'source/sdk/tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in (ROOT/'sdk').iterdir():
        if p.is_file():
            shutil.copy2(p,output/'source/sdk'/p.name)
    shutil.copytree(ROOT/'sdk/src',output/'source/sdk/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(core,output/'source/dohc/infrared')
    shutil.copytree(package/'native',output/'source/native')
    if (ROOT/'sdk/sample').exists():
        shutil.copytree(ROOT/'sdk/sample',output/'source/sdk/sample')
    shutil.copytree(ROOT/'sdk/config',output/'source/sdk/config')
    shutil.copytree(ROOT/'sdk/examples',output/'source/sdk/examples')
    (output/'source/SOURCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    original=dependencies/'python3-pyopencl_2019.1.1-1build1_arm64.deb'
    if original.exists():
        (output/'third-party').mkdir()
        shutil.copy2(original,output/'third-party'/original.name)
    hashes={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*')
            if p.is_file() and stage not in p.parents}
    (output/'SHA256SUMS').write_text(''.join(h+'  '+name+'\n' for name,h in sorted(hashes.items())))
    archive=output.with_name(output.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(output.iterdir()):
            if p!=stage:
                tar.add(p,arcname=output.name+'/'+p.name)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(bundle=str(output),wheel=str(wheel),archive=str(archive),sha256=sha(archive),
                         source_dirty=dirty,offline_dependencies='complete',hardware_validation='pending'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-root',type=Path,required=True)
    parser.add_argument('--dependencies',type=Path,required=True)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    build(args.output,args.native_root,args.dependencies,args.preview)
