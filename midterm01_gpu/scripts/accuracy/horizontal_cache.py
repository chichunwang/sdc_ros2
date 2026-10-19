"""Run-local fine localization map: explicit crop, raw-map identity and payload checks."""
from output_layout import loads_saved_json
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from accuracy.surface_cache import binary_xyz_points
from three_camera_source import sha256


def prepare_fine_map(output, data_root, track, container, bounds, leaf=.3):
    output, data_root = Path(output), Path(data_root)
    root = Path(__file__).resolve().parents[2]
    relative = output.resolve().relative_to((root/'outputs').resolve())
    output.mkdir(parents=True, exist_ok=True)
    values = np.array([bounds['min'], bounds['max']], float)
    if (values.shape != (2, 3) or not np.isfinite(values).all() or np.any(values[1] <= values[0])
            or not np.isfinite(leaf) or leaf <= 0):
        raise ValueError('Invalid fine-map crop/leaf')
    cache = output/f'fine_track{track}.pcd'
    sidecar = cache.with_suffix('.pcd.json')
    source = (data_root/f'Track{track}/map.pcd').resolve()
    identity = {'schema': 'horizontal_fine_map_v1', 'track': track, 'source_map': str(source),
                'map_sha256': sha256(source), 'leaf_m': leaf, 'bounds': bounds}
    if cache.exists() or sidecar.exists():
        if not cache.is_file() or not sidecar.is_file():
            raise ValueError('Incomplete fine-map cache/identity; preserve previous artifacts')
        saved = loads_saved_json(sidecar.read_text())
        if any(saved.get(k) != v for k, v in identity.items()) or saved['cache_sha256'] != sha256(cache):
            raise ValueError('Fine-map cache identity mismatch; preserve previous cache')
        if binary_xyz_points(cache) != saved['points']:
            raise ValueError('Fine-map point count identity mismatch')
        return cache
    container_dir = '/root/outputs/'+relative.as_posix()
    partial = cache.with_suffix('.partial.pcd')
    if partial.exists(): raise ValueError('Incomplete previous fine map; use a new output')
    params = {'input_file': f'/root/data/Track{track}/map.pcd',
              'output_file': container_dir+'/'+partial.name, 'map_voxel_leaf_size': leaf,
              'crop_min': bounds['min'], 'crop_max': bounds['max']}
    config = output/f'fine_track{track}_params.yaml'
    config.write_text(json.dumps({'map_modifier': {'ros__parameters': params}}, indent=2)+'\n')
    command = ['docker', 'exec', container, 'bash', '-c',
        'source /opt/ros/humble/setup.bash && source /root/midterm/install/setup.bash && '
        'exec ros2 run localization map_modified --ros-args --params-file '+shlex.quote(container_dir+'/'+config.name)]
    (output/f'fine_track{track}_command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (output/f'fine_track{track}_build.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Fine-map construction failed; inspect {output}')
    identity.update(points=binary_xyz_points(partial), cache_sha256=sha256(partial))
    partial.rename(cache)
    sidecar.write_text(json.dumps(identity, indent=2)+'\n')
    return cache
