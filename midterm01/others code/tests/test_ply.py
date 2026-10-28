import numpy as np
import pytest

from sdcloc.ply import read_ply


def test_read_ply(tmp_path):
    p = tmp_path / 'a.ply'
    p.write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
                 '0.1 0.2 -0.3 1 2 3\n1.5 2.5 3.5 255 0 128\n')
    xyz, rgb = read_ply(p)
    assert xyz.shape == (2, 3) and rgb.shape == (2, 3)
    assert np.allclose(xyz[1], [1.5, 2.5, 3.5])
    assert rgb[1].tolist() == [255, 0, 128]


def test_read_ply_binary(tmp_path):
    import struct
    p = tmp_path / 'c.ply'
    header = ('ply\nformat binary_little_endian 1.0\ncomment x\nelement vertex 2\nproperty float x\nproperty float y\n'
              'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode()
    body = struct.pack('<fffBBB', 0.5, -1.0, 2.0, 10, 20, 30) + struct.pack('<fffBBB', 3.0, 4.0, 5.0, 1, 2, 3)
    p.write_bytes(header + body)
    xyz, rgb = read_ply(p)
    assert np.allclose(xyz, [[0.5, -1.0, 2.0], [3.0, 4.0, 5.0]])
    assert rgb.tolist() == [[10, 20, 30], [1, 2, 3]]


def test_read_ply_count_mismatch(tmp_path):
    p = tmp_path / 'b.ply'
    p.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n'
                 'end_header\n0 0 0\n1 1 1\n')
    with pytest.raises(ValueError):
        read_ply(p)
