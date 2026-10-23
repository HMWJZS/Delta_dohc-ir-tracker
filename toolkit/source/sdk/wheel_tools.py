"""Small PEP 427 writer and target dependency closure validator."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile
from email.parser import Parser


def write_wheel(output, name, version, tag, files, dependencies=(), requires_python='>=3.8,<3.9'):
    dist = name.replace('-', '_')+'-'+version+'.dist-info'
    files = dict(files)
    metadata = 'Metadata-Version: 2.1\nName: %s\nVersion: %s\nRequires-Python: %s\n' % (name,version,requires_python)
    metadata += ''.join('Requires-Dist: '+dependency+'\n' for dependency in dependencies)
    files[dist+'/METADATA'] = metadata.encode()
    files[dist+'/WHEEL'] = ('Wheel-Version: 1.0\nGenerator: dohc-ir-toolkit\nRoot-Is-Purelib: false\nTag: '+tag+'\n').encode()
    if name == 'dohc-ir-tracker':
        files[dist+'/entry_points.txt'] = b'[console_scripts]\ndohc-ir = dohc_ir_tracker.cli:main\n'
    stream = io.StringIO(newline='')
    rows = csv.writer(stream)
    for path, data in sorted(files.items()):
        rows.writerow([path, 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), len(data)])
    rows.writerow([dist+'/RECORD','',''])
    files[dist+'/RECORD'] = stream.getvalue().encode()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/(name.replace('-', '_')+'-'+version+'-'+tag+'.whl')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026,10,23,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info,data)
    return path


def lock_wheels(directory):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    from packaging.tags import cpython_tags, compatible_tags
    platforms = ['linux_aarch64', 'manylinux2014_aarch64', 'manylinux_2_17_aarch64']
    supported = set(cpython_tags((3, 8), ['cp38'], platforms))
    supported.update(compatible_tags((3, 8), 'cp38', platforms))
    environment = default_environment()
    environment.update(python_version='3.8', python_full_version='3.8.10',
        implementation_name='cpython', implementation_version='3.8.10', sys_platform='linux',
        platform_machine='aarch64', platform_system='Linux', extra='')
    records = {}
    for path in sorted(Path(directory).glob('*.whl')):
        with zipfile.ZipFile(path) as archive:
            metadata = Parser().parsestr(archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode())
        name = canonicalize_name(metadata['Name'])
        _,_,_,tags = parse_wheel_filename(path.name)
        if not tags.intersection(supported):
            raise ValueError('Wheel incompatible with Linux ARM64 CPython 3.8: '+path.name)
        if metadata['Requires-Python'] and Version('3.8.10') not in SpecifierSet(metadata['Requires-Python']):
            raise ValueError('Unsupported Requires-Python: '+path.name)
        if name in records:
            raise ValueError('Duplicate dependency wheel: '+name)
        records[name] = dict(path=path, version=metadata['Version'], requirements=metadata.get_all('Requires-Dist', []))
    for name, record in records.items():
        for raw in record['requirements']:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            target = records.get(canonicalize_name(requirement.name))
            if target is None or Version(target['version']) not in requirement.specifier:
                raise ValueError('Missing/conflicting target dependency for %s: %s' % (name,raw))
    lines = ['# Linux ARM64 / CPython 3.8; generated from actual wheel metadata.']
    for name,r in sorted(records.items()):
        lines.append('%s==%s --hash=sha256:%s' % (name,r['version'],hashlib.sha256(r['path'].read_bytes()).hexdigest()))
    return '\n'.join(lines)+'\n'
