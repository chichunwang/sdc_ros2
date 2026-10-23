"""CUDA exact-neighbor, numerical and rejection contracts; no accuracy claims."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cuda_scale_geometry import CudaKernels,CudaGrid,CudaRegistrar,cuda_normals,_normal_eigh,NORMAL_EIGH_BATCH_SIZE
from map_scale_icp import LayeredScaleMap,ScaleICPConfig,register_frame,normals_at
from cuda_scale_input import load_ply_gpu
from three_camera_calibration import load_ply


@unittest.skipUnless(torch.cuda.is_available(),'Explicit CUDA mechanism tests require a CUDA device')
class CudaGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from accuracy.benchmark import create_run_directory
        cls.output_run=create_run_directory(Path(__file__).resolve().parents[2]/'outputs/run','cuda_scale_mechanisms',track=0)
        cls.kernels=CudaKernels(cls.output_run/'cuda_cache')

    def test_exact_neighbor_ranks_and_out_of_grid(self):
        rng=np.random.default_rng(81);points=rng.uniform(-4,4,(2048,3));query=rng.uniform(-6,6,(100,3))
        query=np.r_[query,[[100.,0,0],[-100.,0,0]]]
        grid=CudaGrid(points,self.kernels)
        for k,radius in [(8,3.),(24,2.5),(8,.1)]:
            distance,ids=grid.query_gpu(torch.tensor(query,device='cuda'),k,radius)
            distance,ids=distance.cpu().numpy(),ids.cpu().numpy();expected,indices=cKDTree(points).query(query,k=k)
            covered=expected<=radius
            np.testing.assert_array_equal(ids[covered],indices[covered])
            np.testing.assert_allclose(distance[covered],expected[covered],atol=1e-12,rtol=0)
            self.assertTrue(np.isinf(distance[~covered]).all())

    def test_ties_use_original_cpu_neighbor_selection(self):
        points=np.vstack([np.zeros((12,3)),np.ones((30,3))])
        grid=CudaGrid(points,self.kernels)
        query=np.zeros((2,3));d,i=grid.query_gpu(torch.tensor(query,device='cuda'),8,3.)
        cd,ci=cKDTree(points).query(query,k=8)
        np.testing.assert_array_equal(i.cpu().numpy(),ci)
        np.testing.assert_array_equal(d.cpu().numpy(),cd)
        self.assertGreater(grid.boundary_refinements,0)

    def test_float64_pca_matches_supported_normal_up_to_sign(self):
        rng=np.random.default_rng(3);points=rng.normal(size=(1024,3));points[:,2]*=.01
        grid=CudaGrid(points,self.kernels)
        normal,quality=cuda_normals(points,grid,np.arange(len(points)))
        expected,confidence=normals_at(points,cKDTree(points),np.arange(len(points)))
        np.testing.assert_allclose(quality,confidence,atol=1e-9,rtol=0)
        supported=(quality>0)&(confidence>0)
        self.assertTrue(supported.any())
        np.testing.assert_allclose(abs(np.sum(normal[supported]*expected[supported],axis=1)),1.,atol=1e-9)

    def test_eigh_workspace_is_bounded_and_preserves_all_matrices(self):
        # Cover the real failed frame's 9070 covariances and an uneven last
        # batch. Distinct eigenvalues make sign-independent comparisons clear.
        rng=np.random.default_rng(653)
        source=torch.tensor(rng.normal(size=(9070,3,3)),device='cuda',dtype=torch.float64)
        covariance=source@source.transpose(1,2)+torch.diag(torch.tensor([1.,2.,4.],device='cuda'))
        expected_eigen,expected_vectors=torch.linalg.eigh(covariance[:NORMAL_EIGH_BATCH_SIZE+1])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated()
        solve=torch.linalg.eigh
        with patch('torch.linalg.eigh',wraps=solve) as bounded:
            eigen,vectors=_normal_eigh(covariance)
        torch.cuda.synchronize()
        sizes=[len(call.args[0]) for call in bounded.call_args_list]
        self.assertEqual(sum(sizes),len(covariance))
        self.assertLessEqual(max(sizes),NORMAL_EIGH_BATCH_SIZE)
        self.assertEqual(sizes[-1],len(covariance)%NORMAL_EIGH_BATCH_SIZE)
        self.assertLess(torch.cuda.max_memory_allocated()-baseline,512*1024**2)
        torch.testing.assert_close(eigen[:len(expected_eigen)],expected_eigen,rtol=1e-12,atol=1e-12)
        alignment=(vectors[:len(expected_vectors)]*expected_vectors).sum(dim=1).abs()
        torch.testing.assert_close(alignment,torch.ones_like(alignment),rtol=0,atol=1e-12)
        # Every matrix, including the final partial batch, has an eigenpair.
        torch.testing.assert_close(covariance@vectors,vectors*eigen[:,None],rtol=1e-11,atol=1e-11)

    def test_registration_and_ground_degeneracy_match_cpu(self):
        from test_map_smoother import room
        points,ground=room();prior=np.eye(4);prior[:3,3]=[.35,-.25,1.62]
        prior[:3,:3]=Rotation.from_euler('xyz',[1.,-.8,1.5],degrees=True).as_matrix()
        config=ScaleICPConfig()
        for label,cloud in [('room',points),('ground',ground)]:
            raw=(cloud-[0,0,1.5])/8.7
            cpu=register_frame(LayeredScaleMap(cloud,config),raw,prior,config)
            gpu=CudaRegistrar(cloud,config,cache_dir=self.output_run/label).register(raw,prior)
            self.assertEqual(cpu['accepted'],gpu['accepted'])
            self.assertEqual(cpu['reasons'],gpu['reasons'])
            np.testing.assert_allclose(gpu['pose'],cpu['pose'],atol=1e-6,rtol=0)
            self.assertAlmostEqual(cpu['scale'],gpu['scale'],places=6)

    def test_no_silent_cpu_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):CudaKernels(self.output_run/'unavailable')

    def test_nonfinite_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'finite'):CudaGrid(np.array([[np.nan,0,0]]),self.kernels)

    def test_voxel_input_preserves_original_rows_at_boundaries(self):
        path=self.output_run/'voxel_boundary_ascii.ply'
        header='ply\nformat ascii 1.0\nelement vertex 6\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
        path.write_text(header+'0.03 0.09 -0.03 0 0 0\n0.03 0.09 -0.03 1 1 1\n0.06 0.12 -0.06 2 2 2\n0.030000000000001 0.09 -0.03 3 3 3\n0 0 0 4 4 4\n-0.03 -0.09 0.03 5 5 5\n')
        for leaf in (0.,.03,.3):
            for maximum in (2,12000):
                np.testing.assert_array_equal(load_ply_gpu(path,max_points=maximum,leaf=leaf),load_ply(path,max_points=maximum,leaf=leaf))


if __name__=='__main__':unittest.main()
