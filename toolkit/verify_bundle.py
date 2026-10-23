#!/usr/bin/env python3
"""Verify every shipped file before installation; no third-party imports."""
import hashlib
from pathlib import Path
import sys


def verify(root):
    root=Path(root).resolve(strict=True)
    count=0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=(root/name).resolve(strict=True)
        path.relative_to(root)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                h.update(chunk)
        if h.hexdigest()!=expected:
            raise ValueError('File hash mismatch: '+name)
        count+=1
    if not count:
        raise ValueError('Empty integrity manifest')
    return count


if __name__=='__main__':
    print('Verified %d files' % verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent))
