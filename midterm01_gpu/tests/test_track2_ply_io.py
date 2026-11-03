"""Track2 reader byte-equivalence; no localization or CUDA."""
from pathlib import Path
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from track3.io import load_ply as legacy
from track2.ply_io import load_ply, _known_ascii_vertex_count


def write_cloud(path, points, *, ascii=True, layout='canonical'):
    specs=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]
    if layout=='double':specs=[(n,'<f8' if n in ('x','y','z') else t) for n,t in specs]
    if layout=='reordered':specs=[specs[i] for i in (3,2,0,4,1,5)]
    a=np.zeros(len(points),dtype=specs)
    for k,n in enumerate(('x','y','z')):a[n]=points[:,k]
    for n in ('red','green','blue'):a[n]=np.arange(len(a))%256
    PlyData([PlyElement.describe(a,'vertex')],text=ascii).write(str(path))


@pytest.mark.parametrize('count',[0,1,37,1000])
@pytest.mark.parametrize('layout',['canonical','double','reordered'])
@pytest.mark.parametrize('ascii',[True,False])
def test_identical_values_order_dtype_and_bytes(tmp_path,count,layout,ascii):
    points=np.random.default_rng(71).normal(size=(count,3))
    if count>3:points[:3]=[[np.nan,0,1],[2,np.inf,1],[0,-0.,1]]
    f=tmp_path/'cloud.ply';write_cloud(f,points,ascii=ascii,layout=layout)
    a,b=legacy(f),load_ply(f)
    assert a.dtype==b.dtype==np.float64 and a.shape==b.shape
    assert a.tobytes()==b.tobytes()


def test_decimal_boundaries_and_signed_zero(tmp_path):
    # Includes representable-f32 boundaries, subnormals and values above 2**24.
    values=['-0','0','1.0000000596046447','1.0000000596046448','16777217',
            '1.17549435e-38','1.401298464324817e-45','-1.401298464324817e-45']
    f=tmp_path/'edge.ply';header='ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'%len(values)
    f.write_text(header+''.join(f'{v} {v} {v} 1 2 3\n' for v in values))
    assert legacy(f).tobytes()==load_ply(f).tobytes()


@pytest.mark.parametrize('payload',['1 2 3 4 5\n','1 2 bad 4 5 6\n','1 2 3 4 5 6\n1 2 3 4 5 6\n'])
def test_malformed_input_retains_legacy_failure(tmp_path,payload):
    f=tmp_path/'bad.ply';f.write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'+payload)
    # The third payload is valid; preserve both success and failure behavior.
    try:expected=legacy(f)
    except Exception as error:
        with pytest.raises(type(error)):load_ply(f)
    else:assert load_ply(f).tobytes()==expected.tobytes()
