import csv
from pathlib import Path
import unittest
import numpy as np
from calibrated_rig_geometry import balanced_indices,voxel_indices,rectify
from rig_geometry_accel import GeometryAccelerator,CachedRectifier


class AccelerationTests(unittest.TestCase):
    def compare(self,points,cameras,confidence,budget=600,backend='cpu'):
        expected,report=balanced_indices(points,cameras,confidence,budget)
        engine=GeometryAccelerator(backend)
        actual,actual_report=engine.balanced_indices(points,cameras,confidence,budget)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual_report,report)
        return engine

    def planes(self):
        x,y=np.meshgrid(np.linspace(-5,5,35),np.linspace(-5,5,35))
        ground=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-2.)))
        u,z=np.meshgrid(np.linspace(-4,4,30),np.linspace(-1,5,25))
        wall=np.column_stack((u.ravel(),u.ravel(),z.ravel()))
        points=np.vstack((ground,wall,wall+[1,0,0]))
        cameras=np.repeat([0,1,2],[len(ground),len(wall),len(wall)])
        return points,cameras,np.ones(len(points))

    def test_cpu_batch_planes_axis_ties_and_confidence_ties(self):
        points,cameras,confidence=self.planes()
        engine=self.compare(points,cameras,confidence)
        self.assertGreater(engine.stats['scalar_boundary_refinements'],0)

    def test_spatial_round_robin_random_geometry_and_small_budget(self):
        rng=np.random.default_rng(5)
        points=rng.normal(size=(4000,3))*3
        cameras=rng.integers(0,3,len(points));confidence=rng.integers(1,4,len(points)).astype(float)
        self.compare(points,cameras,confidence,6)
        self.compare(points,cameras,confidence,900)

    def test_rejected_geometry_and_sparse_case(self):
        self.compare(np.zeros((30,3)),np.zeros(30,dtype=int),np.ones(30))
        self.compare(np.array([[0.,0,0],[1,0,0]]),np.zeros(2,dtype=int),np.ones(2))

    def test_cached_rectification_preserves_image_k_and_mask(self):
        root=Path(__file__).resolve().parents[3]/'midterm01/data'
        if not root.exists():root=Path('/root/data')
        if not root.exists():self.skipTest('Real source data unavailable')
        cache=CachedRectifier()
        for track,idx in [(1,1505),(2,1661)]:
            with (root/f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            for tag in ('f','fl','fr'):
                camera_root=root/f'Track{track}/Track{track}_images'
                image=camera_root/f'Track{track}_{tag}_images/{idx:06d}_ts_{rows[idx][tag+"_ts"]}.jpg'
                calibration=camera_root/f'camera_info/gige_100_{tag}_hdr_camera_info.yaml'
                mask=camera_root/'camera_info/gige_100_f_hdr_mask.png' if tag=='f' else None
                expected=rectify(image,calibration,exclusion_mask=mask)
                for _ in range(2):
                    for a,b in zip(expected,cache(image,calibration,exclusion_mask=mask)):np.testing.assert_array_equal(a,b)

    def test_cuda_integer_voxel_keys_and_pca_match_reference(self):
        try:
            import torch
        except ImportError:self.skipTest('Torch not installed in ROS environment')
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        engine=GeometryAccelerator('cuda')
        points=np.array([[.3,0,0],[.3-1e-15,0,0],[.3+1e-15,0,0],[-.3,1,2],[-.3,1,2]])
        np.testing.assert_array_equal(engine.voxel_indices(points),voxel_indices(points))
        p,c,q=self.planes();self.compare(p,c,q,backend='cuda')


if __name__=='__main__':unittest.main()
