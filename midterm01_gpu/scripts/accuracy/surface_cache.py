"""Build a run-local height map from raw data; never depend on historical runs."""

import csv
import json
import math
from pathlib import Path
import shlex
import subprocess

from three_camera_source import sha256


def height_band_bounds(prior_file):
    with Path(prior_file).open() as stream:
        centers = [tuple(float(row[k]) for k in ('x', 'y', 'z')) for row in csv.DictReader(stream)]
    if not centers or any(not math.isfinite(v) for row in centers for v in row):
        raise ValueError('Height map requires nonempty finite sensor priors')
    # Cover every 22 m query plus its PCA neighborhood; same band as the model.
    return {
        'min': [min(p[0] for p in centers)-25., min(p[1] for p in centers)-25., min(p[2] for p in centers)-6.],
        'max': [max(p[0] for p in centers)+25., max(p[1] for p in centers)+25., max(p[2] for p in centers)-.3],
    }


def binary_xyz_points(path):
    """Reject absent, empty, compressed or truncated outputs before certification."""
    header = {}
    with Path(path).open('rb') as stream:
        for _ in range(32):
            line = stream.readline(1024).decode('ascii').strip()
            if line and not line.startswith('#'):
                key, *values = line.split()
                header[key] = values
                if key == 'DATA':
                    break
        else:
            raise ValueError('Missing PCD header')
        expected = {'FIELDS': ['x', 'y', 'z'], 'SIZE': ['4']*3,
                    'TYPE': ['F']*3, 'COUNT': ['1']*3, 'DATA': ['binary']}
        if any(header.get(k) != v for k, v in expected.items()):
            raise ValueError('Height map must be binary float32 XYZ PCD')
        points = int(header['POINTS'][0])
        if points <= 0 or int(header['WIDTH'][0])*int(header['HEIGHT'][0]) != points:
            raise ValueError('Invalid PCD point count')
        if Path(path).stat().st_size-stream.tell() != points*12:
            raise ValueError('Truncated or inconsistent PCD payload')
    return points


def prepare_height_map(run, data_root, track, container):
    """Publish a verified cache/sidecar only after map_modified succeeds."""
    run, data_root = Path(run), Path(data_root)
    cache = run / f'cache_height_track{track}_leaf03.pcd'
    partial = cache.with_name(cache.stem+'.partial.pcd')
    sidecar = cache.with_suffix(cache.suffix+'.json')
    if any(p.exists() for p in (cache, partial, sidecar)):
        raise ValueError('Height map output already exists; use a new run')
    bounds = height_band_bounds(run/'sensor_prior.csv')
    container_run = '/root/outputs/run/'+run.name
    params = {'input_file': f'/root/data/Track{track}/map.pcd',
              'output_file': container_run+'/'+partial.name, 'map_voxel_leaf_size': .3,
              'crop_min': bounds['min'], 'crop_max': bounds['max']}
    (run/'height_map_params.json').write_text(json.dumps(params, indent=2)+'\n')
    (run/'height_map_params.yaml').write_text(json.dumps({'map_modifier': {'ros__parameters': params}}, indent=2)+'\n')
    command = ['docker', 'exec', container, 'bash', '-c',
               'source /opt/ros/humble/setup.bash && source /root/midterm/install/setup.bash && '
               'exec ros2 run localization map_modified --ros-args --params-file '
               + shlex.quote(container_run+'/height_map_params.yaml')]
    (run/'height_map_command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (run/'height_map_preparation.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Height map preprocessing exited {result.returncode}; inspect height_map_preparation.log')
    points = binary_xyz_points(partial)
    source = (data_root/f'Track{track}/map.pcd').resolve()
    identity = {'track': track, 'source_map': str(source), 'map_sha256': sha256(source),
                'map_voxel_leaf_size': .3, 'height_band_bounds': bounds,
                'cache_sha256': sha256(partial), 'points': points,
                'sensor_prior_sha256': sha256(run/'sensor_prior.csv'),
                'derivation': 'map_modified current-run prior XY+/-25m and nominal Z[-6,-0.3] crop, then checked PCL 0.3m VoxelGrid'}
    partial.rename(cache)
    sidecar.write_text(json.dumps(identity, indent=2)+'\n')
    return cache
