import numpy as np

'''
Make sure you set the parameters.
'''
class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=0,
        fitness_thresh=0,
        max_candidates=25,
    ):
        """
        icp: ICP instance (small_gicp wrapper)
        """

        self.icp = icp

        # ===== parameter =====
        self.dist_thresh = dist_thresh

        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates

        # ===== state =====
        self.keyframes = []
        self.poses = []
        
        self.last_loop_accum = 0.0

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose):
        # local frame points
        self.keyframes.append(pts)
        
        # world frame poses
        self.poses.append(pose)


    # ==========================
    #    Candidate selection
    # ==========================
    def find_candidates(self, curr_idx):
        candidates = []

        """
        [TODO] 
        
            Choose which candidates to verify loop closure,
            You could base  on distance, time interval 
            
            return: index of candidates
        """

        return candidates

    # =========================
    #       ICP matching
    # =========================
    def match(self, curr_idx, candidates, submap_mgr):
        '''
        curr_idx: current
        candidates: candidates that you choose to do loop closure detection with current frame
        submap_mgr: use as submap_mgr.submaps[idx] to get idx-th submap points(those are in world frame)
        '''
        

        best_score = np.inf
        best_idx = None
        best_T = None

        for idx in candidates:
            """
            TODO: 
            1. Perform scan matching between curr_idx and candidate idx.
            2. Evaluate the match quality.
            3. Update best_score and best_T
            
            4. If valid loop is found, return corresponding results,
               otherwise return None.
            """
        
        return best_idx, best_T, best_score

    # =========================
    #        Pose Update
    # =========================
    def update_poses(self, new_kf_poses):
        """ Sync poses and recalculate accumulated distances after PGO """
        # Update poses list
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = new_kf_poses[i].copy()
    
    # =========================
    #            API
    # =========================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        # Could block some too-early detections
        if curr_idx < 30:
            return None

        candidates = self.find_candidates(curr_idx)
        result = self.match(curr_idx, candidates, submap_mgr) 
        return result