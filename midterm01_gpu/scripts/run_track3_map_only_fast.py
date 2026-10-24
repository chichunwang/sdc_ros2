#!/usr/bin/env python3
"""Same causal Track3 v2 model, with fused FP64 reductions/shared data/streams."""
from pathlib import Path
import sys

# The staged and installed entry points use the same shared reference runtime.
location=Path(__file__).resolve().parent
for candidate in (location.parent,Path.cwd()):
    if (candidate/'scripts/track3_registration.py').is_file():
        sys.path.insert(1,str(candidate/'scripts'));break
if (location.parent/'staged_reference').is_dir():
    sys.path.insert(1,str(location.parent/'staged_reference'))

# Load the tracker hooks before accuracy.benchmark prepends the live scripts
# directory to sys.path. Installed and staged execution must resolve alike.
from track3_map_only_fast import METHOD, MapOnlyTracker
from track3_registration_fast import FastGeometry, FastMapRegistrar
if not hasattr(MapOnlyTracker.__mro__[1],'_map_candidates'):
    raise RuntimeError('Shared tracker lacks execution hooks; refusing a serial legacy tracker')
from run_track3_map_only import main as reference_main

BACKEND='cuda_float64_fused_pairs_statistics_streams_ndt_icp_v2_fast_no_imu'


def main(argv=None):
    extra=()
    if (location.parent/'staged_reference').is_dir():
        extra=tuple(location.glob('*.py'))+tuple((location.parent/'staged_reference').glob('*.py'))
    return reference_main(argv,method=METHOD,backend=BACKEND,geometry_type=FastGeometry,
        registrar_type=FastMapRegistrar,tracker_type=MapOnlyTracker,
        entrypoint=Path(__file__),extra_sources=extra)


if __name__=='__main__':
    raise SystemExit(main())
