import small_gicp as sg
import numpy as np

class ICP:
    def __init__(self, voxel_size=0.25):
        self.voxel_size = voxel_size

    def align(self, src_pts, tgt_pts, init_T=np.eye(4)):
        T = None
        score = None
        
        """
        TODO:
        Choose your Scan-matching ALGO and set parameters
        
        """

        return np.asarray(T), score
