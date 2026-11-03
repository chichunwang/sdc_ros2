import numpy as np

from sdcloc.mapio import crop_box, voxel_downsample


def test_voxel_downsample_centroids():
    xyz = np.array([[0.1, 0.1, 0.1], [0.3, 0.2, 0.1], [2.1, 0.0, 0.0], [-0.4, 0.0, 0.0]])
    ds, ex, counts = voxel_downsample(xyz, 1.0, extra=np.array([1.0, 3.0, 5.0, 7.0]))
    assert len(ds) == 3
    i = np.argmin(np.abs(ds[:, 0] - 0.2))
    assert np.allclose(ds[i], [0.2, 0.15, 0.1]) and counts[i] == 2 and ex[i, 0] == 2.0


def test_crop_box():
    xyz = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 9], [-3, 2, 1]], dtype=float)
    m = crop_box(xyz, (0, 0, 0), half_xy=4, half_z=2)
    assert m.tolist() == [True, False, False, True]
