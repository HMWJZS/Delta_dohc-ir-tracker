#!/usr/bin/env python3
"""Prepare ARM64 wheels locally; does not contact or alter a DOHC."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from wheel_tools import write_wheel, lock_wheels

DEB = 'python3-pyopencl_2019.1.1-1build1_arm64.deb'
URL = 'https://ports.ubuntu.com/ubuntu-ports/pool/universe/p/pyopencl/'+DEB
SHA = '66e5d560593dc8c3d9b242b0c10d3859cd3f237605171a275564462e7a3b7974'


def prepare(output, download=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if download:
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:',
            '--platform','manylinux2014_aarch64','--platform','manylinux_2_17_aarch64','--platform','any',
            '--python-version','38','--implementation','cp','--abi','cp38','--abi','abi3','--abi','none',
            '--dest',str(output),'-r',str(Path(__file__).with_name('requirements-arm64.in'))],check=True)
    original = output/DEB
    if not original.exists():
        if not download:
            raise FileNotFoundError(original)
        original.write_bytes(urllib.request.urlopen(URL,timeout=60).read())
    if hashlib.sha256(original.read_bytes()).hexdigest() != SHA:
        raise ValueError('Ubuntu PyOpenCL source package hash mismatch')
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['dpkg-deb','-x',str(original),tmp],check=True)
        site = Path(tmp)/'usr/lib/python3/dist-packages'
        files = {p.relative_to(site).as_posix():p.read_bytes() for p in (site/'pyopencl').rglob('*') if p.is_file()}
        for p in (Path(tmp)/'usr/share/doc/python3-pyopencl').rglob('*'):
            if p.is_file():
                files['pyopencl-2019.1.1.dist-info/licenses/'+p.name] = p.read_bytes()
        files['pyopencl-2019.1.1.dist-info/ubuntu-provenance.json'] = json.dumps(dict(
            url=URL,sha256=SHA,transformation='wheel container only; pyopencl files and ELF unchanged',
            board_validation='pending', required_external_library='libOpenCL.so.1 from approved Mali ICD')).encode()
        write_wheel(output,'pyopencl','2019.1.1','cp38-cp38-linux_aarch64',files,
            ['numpy>=1.16,<1.24','pytools>=2017.6','appdirs>=1.4','decorator','six>=1.9','setuptools'])
    lock = lock_wheels(output)
    (output/'requirements.lock').write_text(lock)
    return lock


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-download',action='store_true')
    args=parser.parse_args()
    print(prepare(args.output,not args.no_download))
