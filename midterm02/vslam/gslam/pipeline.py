"""視覺引導的 GPU 幾何 SLAM 主流程。

每幀：載入 → GPU 前處理 → 影像特徵＋GPU 比對 → PnP 相對位姿（視覺初值）→ GPU ICP/GICP 對局部子圖配準
→ 品質閘門（不通過就換初值／放寬對應距離重試，仍失敗則沿用預測並標記）→ keyframe → 視覺閉環＋點雲驗證 → pose graph。

狀態說明：
- self.T_prev：上一幀在「目前地圖座標」的 pose；閉環最佳化後會跟著修正。
- self.odom：純 odometry 鏈（每幀只累乘當幀的相對運動，永遠不被閉環修改），用來輸出 poses_odometry.csv、
  當作 pose graph 的 odometry 量測，以及最後把 keyframe 的修正分配給所有幀。
"""
import time

import numpy as np
import cupy as cp

from . import se3
from .gpu_cloud import TargetModel, sync, voxel_downsample
from .gpu_registration import register, source_covariances
from .preprocess import preprocess
from . import visual as vis_mod
from .scale import ground_height


class Keyframe:
    def __init__(self, kf_id, idx, T, scale, src, src_cov, feats, pts3d, valid3d):
        self.id, self.idx, self.T, self.scale = kf_id, idx, T.copy(), float(scale)
        self.src, self.src_cov = src, src_cov          # GPU：降採樣後的相機座標點（與 GICP 共變異數）
        self.feats, self.pts3d, self.valid3d = feats, pts3d, valid3d


class Slam:
    def __init__(self, dataset, params, debug_dir=None, log=print, viz=None):
        self.ds, self.p, self.debug_dir, self.log = dataset, params, debug_dir, log
        self.viz = viz             # RViz 串流（VizStream）；None = 不顯示
        self.frontend = vis_mod.make_frontend(params['visual']) if params['visual']['enabled'] else None
        self.keyframes = []
        self.loops = []            # 已接受的閉環邊：(match_kf, query_kf, T_rel)
        self.loop_rows = []        # 所有候選的紀錄
        self.records = []
        self.odom = []
        self.T_prev = np.eye(4)
        self.velocity = np.eye(4)
        self.scale_prev = 1.0
        self.prev_feats = self.prev_pts3d = self.prev_valid = self.prev_img = None
        self.model = None
        self.frames_since_kf = 0
        self.last_loop_kf = -10 ** 9
        self.bank = None
        self.bank_owner = None
        self.bank_upto = 0
        self.optimizations = 0
        self.h_ref = None
        self.cloud_scale_prev = 1.0

    # ------------------------------------------------------------ 子圖
    def _world_points(self, kf):
        R = cp.asarray(kf.T[:3, :3]); t = cp.asarray(kf.T[:3, 3])
        return kf.scale * (kf.src @ R.T) + t

    def _build_model(self, kf_ids):
        cfg, reg = self.p['submap'], self.p['registration']
        pts = voxel_downsample(cp.concatenate([self._world_points(self.keyframes[k]) for k in kf_ids]), cfg['voxel_m'])
        if len(pts) > cfg['max_points']:
            pts = pts[cp.linspace(0, len(pts) - 1, cfg['max_points']).astype(cp.int64)]
        return TargetModel(pts, reg['index_cell_m'], reg['normal_radius_m'], reg['min_neighbors'], reg['planar_eps'])

    def _rebuild_local_model(self):
        n = len(self.keyframes)
        try:
            self.model = self._build_model(range(max(0, n - self.p['submap']['window_keyframes']), n))
        except ValueError as exc:  # 幾何不足：保留舊子圖
            self.log('警告：子圖重建失敗（%s），沿用舊子圖' % exc)

    # ------------------------------------------------------------ 配準與閘門
    def _prepare_source(self, cloud):
        reg = self.p['registration']
        if reg['method'] != 'gicp':
            return cloud, None
        valid, cov = source_covariances(cloud, reg['normal_radius_m'], reg['min_neighbors'], reg['planar_eps'])
        return cp.ascontiguousarray(cloud[valid]), cp.ascontiguousarray(cov[valid])

    def _fusion(self, visual, T_visual):
        """視覺先驗的權重（每個對應）。

        配準代價 Σ e_i²（公尺²）若每點雜訊 σ_p、但誤差高度相關只有 n_eff 個獨立量測，其資訊量相當於
        (n_eff/n)·Σ e_i²/σ_p²；視覺位姿的資訊量是 |ξ_v|²/σ_t² + |ξ_ω|²/σ_r²。把兩者同乘 σ_p²·n/n_eff，
        視覺項的係數就是 n·(σ_p/σ)²/n_eff —— 也就是下面回傳的「每個對應的權重」。
        """
        f = self.p['fusion']
        if not f['enabled'] or visual is None or not visual['ok']:
            return None, None
        grow = max(1.0, np.sqrt(f['reference_inliers'] / max(visual['pnp_inliers'], 1)))
        wt = (f['point_sigma_m'] / (f['visual_trans_sigma_m'] * grow)) ** 2 / f['effective_points']
        wr = (f['point_sigma_m'] / (np.deg2rad(f['visual_rot_sigma_deg']) * grow)) ** 2 / f['effective_points']
        return T_visual, (wt, wr)

    def _register(self, src, src_cov, model, T_init, stages, scale_init, prior_T=None, prior_weights=None):
        reg = self.p['registration']
        return register(src, model, T_init, method=reg['method'], stages=stages, robust_k=reg['robust_k_m'],
                        src_cov=src_cov, scale_init=scale_init, estimate_scale=reg['estimate_scale'],
                        scale_prior_weight=reg['scale_prior_weight'],
                        min_correspondences=reg['min_correspondences'], prior_T=prior_T, prior_weights=prior_weights)

    def _gate(self, result, T_init, visual, prior_name):
        """回傳不通過的原因清單（空清單 = 接受）。"""
        g = self.p['gate']
        reasons = []
        if result['reason'] in ('insufficient_correspondences', 'singular_system'):
            reasons.append(result['reason'])
        if result['fitness'] < g['min_fitness']:
            reasons.append('low_fitness')
        if not result['inlier_rmse'] <= g['max_inlier_rmse_m']:
            reasons.append('high_rmse')
        if prior_name == 'visual':
            # 視覺初值本身是量測：配準結果不應離它太遠
            dt, dr = se3.relative_error(T_init, result['T'])
            if dt > g['max_translation_from_prior_m']:
                reasons.append('translation_jump')
            if np.rad2deg(dr) > g['max_rotation_from_prior_deg']:
                reasons.append('rotation_jump')
        # 等速預測在轉彎時不準，不拿它當標準；改檢查單幀運動量是否在車輛物理上合理的範圍
        step = se3.inv(self.T_prev) @ result['T']
        if np.linalg.norm(step[:3, 3]) > g['max_step_translation_m']:
            reasons.append('step_translation_limit')
        if np.rad2deg(se3.rotation_angle(step[:3, :3])) > g['max_step_rotation_deg']:
            reasons.append('step_rotation_limit')
        eig = result['normalized_min_eigenvalue']
        if eig is not None and eig < g['min_normalized_eigenvalue']:
            reasons.append('degenerate_geometry')
        if visual is not None and visual['ok']:
            rel = se3.inv(self.T_prev) @ result['T']
            angle = se3.rotation_angle(rel[:3, :3].T @ visual['T_ref_cur'][:3, :3])
            if np.rad2deg(angle) > g['max_visual_rotation_disagreement_deg']:
                reasons.append('visual_rotation_disagreement')
        return reasons

    # ------------------------------------------------------------ 每幀
    def process(self, local, i):
        p = self.p
        t_start = time.perf_counter()
        xyz, rgb = self.ds.load_cloud(i)
        image = self.ds.load_image(i) if (self.frontend is not None or self.viz is not None) else None
        # ---- 逐幀尺度正規化（相機離地高度）：之後所有幾何（配準、PnP 的 3D 點）都用正規化後的點雲
        height, cloud_scale = '', 1.0
        if p['scale']['mode'] == 'ground_height':
            h, _ = ground_height(xyz, p['scale'], seed=p['seed'] + local)
            if h is not None:
                height = h
                if self.h_ref is None:
                    self.h_ref = h
                cloud_scale = self.h_ref / h
                if abs(cloud_scale / self.cloud_scale_prev - 1) > p['scale']['max_change_ratio']:
                    cloud_scale = self.cloud_scale_prev
            else:
                cloud_scale = self.cloud_scale_prev
            self.cloud_scale_prev = cloud_scale
            xyz = xyz * cloud_scale
        load_ms = (time.perf_counter() - t_start) * 1000

        t0 = time.perf_counter()
        cloud, pstats = preprocess(xyz, rgb, p['preprocess'])
        src, src_cov = self._prepare_source(cloud) if len(cloud) >= 100 else (cloud, None)
        sync()
        preprocess_ms = (time.perf_counter() - t0) * 1000

        feats = pts3d = valid3d = None
        feature_ms = 0.0
        if image is not None and self.frontend is not None:
            t0 = time.perf_counter()
            feats = self.frontend.extract(image)
            pix = vis_mod.PixelCloud(xyz, self.ds.K, self.ds.size, p['visual']['assoc_stride_px'], p['visual']['max_depth_m'])
            pts3d, valid3d = pix.lookup(feats.kpts)
            sync()
            feature_ms = (time.perf_counter() - t0) * 1000

        rec = dict(idx=local, timestamp_ns=self.ds.timestamps[i], backend='cuda:' + p['registration']['method'],
                   load_ms=load_ms, preprocess_ms=preprocess_ms, registration_ms=0.0, total_ms=0.0, is_keyframe=0,
                   fitness='', inlier_rmse='', points_raw=pstats['raw'], points_used=int(len(src)),
                   keypoints=0 if feats is None else len(feats), matches=0, essential_inliers=0, pnp_inliers=0,
                   visual_inlier_ratio=0.0, visual_reprojection_px='', visual_ok=0, prior_used='', registration_method='',
                   attempts=0, converged='', iterations='', correspondences='', normalized_min_eigenvalue='',
                   gate_reasons='', tracking_status='', loop_status='', estimated_scale=1.0, visual_scale_ratio='',
                   feature_ms=feature_ms, matching_ms=0.0, step_translation_m=0.0, step_rotation_deg=0.0,
                   ground_height=height, cloud_scale=cloud_scale, vg_translation_m='', vg_rotation_deg='',
                   visual_mode='', flow_px='', visual_length_m='', visual_length_mad_m='', pnp_length_m='')

        visual = matches = None
        if local == 0:
            T, scale, status = np.eye(4), 1.0, 'init'
            self.odom.append(np.eye(4))
            rec.update(tracking_status=status, prior_used='identity', registration_method='none')
            make_kf = True
        else:
            # ---- 視覺相對位姿
            if feats is not None and self.prev_feats is not None:
                t0 = time.perf_counter()
                matches = self.frontend.match(self.prev_feats, feats)
                sync()
                rec['matching_ms'] = (time.perf_counter() - t0) * 1000
                visual = vis_mod.estimate_relative_pose(self.prev_feats.kpts, feats.kpts, matches, self.prev_pts3d,
                                                        self.prev_valid, self.ds.K, p['visual'], pts3d, valid3d,
                                                        seed=p['seed'] + local)
                rec.update(matches=visual['matches'], essential_inliers=visual['essential_inliers'],
                           pnp_inliers=visual['pnp_inliers'], visual_inlier_ratio=visual['inlier_ratio'],
                           visual_reprojection_px='' if visual['reprojection_px'] is None else visual['reprojection_px'],
                           visual_ok=int(visual['ok']), visual_mode=visual['mode'], flow_px=visual['flow_px'],
                           **{k: ('' if visual[s] is None else visual[s]) for k, s in (
                               ('visual_length_m', 'length_m'), ('visual_length_mad_m', 'length_mad_m'),
                               ('pnp_length_m', 'pnp_length_m'))},
                           visual_scale_ratio='' if visual['scale_ratio'] is None else visual['scale_ratio'])
                every = p['visual']['debug_every']
                if self.debug_dir is not None and every and local % every == 0 and self.prev_img is not None:
                    vis_mod.draw_matches(self.prev_img, image, self.prev_feats.kpts, feats.kpts, matches,
                                         visual['essential_mask'] if visual['mode'] == 'essential' else visual['pnp_mask'],
                                         self.debug_dir / 'features' / ('%05d_%s.jpg' % (local, rec['timestamp_ns'])),
                                         text='matches %d  E-inliers %d  PnP-inliers %d' % (
                                             visual['matches'], visual['essential_inliers'], visual['pnp_inliers']))

            # ---- 初值候選
            priors = {}
            priors['motion'] = se3.normalize(self.T_prev @ self.velocity)
            priors['identity'] = self.T_prev.copy()
            if visual is not None and visual['ok']:
                Tv = visual['T_ref_cur'].copy()
                Tv[:3, 3] *= self.scale_prev   # PnP 的平移是上一幀點雲的尺度，換成地圖尺度
                priors['visual'] = se3.normalize(self.T_prev @ Tv)
            want = p['registration']['prior']
            primary = want if want in priors else 'motion'
            order = [(primary, p['registration']['stages'])]
            for alt in ('visual', 'motion'):
                if alt != primary and alt in priors and want != 'identity':
                    order.append((alt, p['registration']['recovery_stages']))
            if len(order) == 1:
                order.append((primary, p['registration']['recovery_stages']))

            # ---- GPU 配準（依序嘗試，第一個通過閘門的就採用）
            t0 = time.perf_counter()
            accepted = None
            reasons = ['no_source_points']
            result = None
            if len(src) >= p['registration']['min_correspondences'] and self.model is not None:
                for attempt, (name, stages) in enumerate(order):
                    fuse_T, fuse_w = self._fusion(visual, priors.get('visual')) if name == 'visual' else (None, None)
                    result = self._register(src, src_cov, self.model, priors[name], stages, self.scale_prev, fuse_T, fuse_w)
                    reasons = self._gate(result, priors[name], visual, name)
                    rec['attempts'] = attempt + 1
                    if not reasons:
                        accepted = (name, attempt)
                        break
            sync()
            rec['registration_ms'] = (time.perf_counter() - t0) * 1000
            if result is not None:
                rec.update(fitness=result['fitness'], inlier_rmse=result['inlier_rmse'], converged=int(result['converged']),
                           iterations=result['iterations'], correspondences=result['correspondences'],
                           normalized_min_eigenvalue='' if result['normalized_min_eigenvalue'] is None
                           else result['normalized_min_eigenvalue'])
            if accepted is not None:
                T, scale = result['T'], result['scale']
                status = 'ok' if accepted[1] == 0 else 'recovered'
                rec.update(prior_used=accepted[0], registration_method='%s+%s%s' % (
                    accepted[0], p['registration']['method'], '' if accepted[1] == 0 else '(recovery)'))
            else:
                # 全部失敗：沿用預測（視覺可靠就用視覺，否則等速），不讓 pose 跳動，也不清掉運動狀態
                T, scale, status = priors[primary], self.scale_prev, 'tracking_failure'
                rec.update(prior_used=primary, registration_method='predict_only', gate_reasons='|'.join(reasons))
            rec['tracking_status'] = status
            rel = se3.inv(self.T_prev) @ T
            if 'visual' in priors:   # 配準結果與影像估計的差（自我一致性指標）
                dt, dr = se3.relative_error(priors['visual'], T)
                rec.update(vg_translation_m=dt, vg_rotation_deg=float(np.rad2deg(dr)))
            self.odom.append(se3.normalize(self.odom[-1] @ rel))
            self.velocity = rel
            rec['step_translation_m'] = float(np.linalg.norm(rel[:3, 3]))
            rec['step_rotation_deg'] = float(np.rad2deg(se3.rotation_angle(rel[:3, :3])))
            kcfg = p['keyframe']
            dt, dr = se3.relative_error(self.keyframes[-1].T, T)
            self.frames_since_kf += 1
            moved = dt > kcfg['translation_m'] or np.rad2deg(dr) > kcfg['rotation_deg']
            make_kf = (moved and status != 'tracking_failure') or self.frames_since_kf >= kcfg['max_frames_between']
            make_kf = make_kf and len(src) >= 100

        rec['estimated_scale'] = scale
        self.T_prev, self.scale_prev = T, scale
        every = p['debug']['cloud_every']
        if self.debug_dir is not None and every and local % every == 0 and self.model is not None and len(src):
            from .outputs import save_cloud_topdown
            R = cp.asarray(T[:3, :3]); t = cp.asarray(T[:3, 3])
            save_cloud_topdown(self.debug_dir / 'cloud' / ('%05d_%s.png' % (local, rec['timestamp_ns'])),
                               cp.asnumpy(self.model.points), cp.asnumpy(scale * (src @ R.T) + t),
                               'frame %d  status=%s' % (local, rec['tracking_status']))

        if self.viz is not None:
            vcfg = p['viz']
            vx, vc = self.viz.color_cloud(xyz, rgb, vcfg['max_depth_m'], vcfg['frame_point_step'])
            annotated = None
            if image is not None:
                mask = None if visual is None else (visual['essential_mask'] if visual['mode'] == 'essential'
                                                    else visual['pnp_mask'])
                lines = ['frame %d  %s  kf %d  loops %d' % (local, rec['tracking_status'], len(self.keyframes), len(self.loops)),
                         'visual: %s  matches %s  inliers %s' % (rec['visual_mode'] or '-', rec['matches'], rec['essential_inliers']),
                         'registration: %s  fitness %s  rmse %s' % (rec['registration_method'] or '-',
                                                                    ('%.2f' % rec['fitness']) if rec['fitness'] != '' else '-',
                                                                    ('%.3f' % rec['inlier_rmse']) if rec['inlier_rmse'] != '' else '-'),
                         'step %.2f m  %.2f deg' % (rec['step_translation_m'], rec['step_rotation_deg'])]
                annotated = self.viz.annotate(image, feats, self.prev_feats, matches, mask, lines)
            self.viz.frame(rec, T, self.odom[-1], image, annotated, vx * scale, vc)

        if make_kf:
            kf = Keyframe(len(self.keyframes), local, T, scale, src, src_cov, feats, pts3d, valid3d)
            self.keyframes.append(kf)
            self.frames_since_kf = 0
            rec['is_keyframe'] = 1
            self._rebuild_local_model()
            if local > 0:
                rec['loop_status'] = self._detect_loop(kf)
                if rec['loop_status'] == 'accepted':
                    self.optimize()
                    self._rebuild_local_model()
                    self.T_prev = self.keyframes[-1].T.copy()
            if self.viz is not None:
                kx, kc = self.viz.color_cloud(xyz, rgb, p['viz']['max_depth_m'], p['viz']['keyframe_point_step'])
                self.viz.keyframe(kf.id, kx * scale, kc)
                sub = None
                if self.model is not None:
                    pts = self.model.points
                    sub = cp.asnumpy(pts[::max(1, len(pts) // p['viz']['submap_max_points'])])
                self.viz.graph([k.T for k in self.keyframes], [(c, q) for c, q, _ in self.loops], sub)
        self.prev_feats, self.prev_pts3d, self.prev_valid, self.prev_img = feats, pts3d, valid3d, image
        rec['total_ms'] = (time.perf_counter() - t_start) * 1000
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------ 閉環
    def _update_bank(self, upto):
        """把編號 < upto 的 keyframe 描述子加入檢索庫（GPU）。"""
        new_desc, new_owner = [], []
        for k in range(self.bank_upto, upto):
            f = self.keyframes[k].feats
            if f is not None and len(f):
                new_desc.append(f.desc)
                new_owner.append(cp.full(len(f), k, dtype=cp.int64))
        self.bank_upto = max(self.bank_upto, upto)
        if new_desc:
            parts_d = ([self.bank] if self.bank is not None else []) + new_desc
            parts_o = ([self.bank_owner] if self.bank_owner is not None else []) + new_owner
            self.bank, self.bank_owner = cp.concatenate(parts_d), cp.concatenate(parts_o)

    def _detect_loop(self, kf):
        cfg = self.p['loop']
        if not cfg['enabled'] or kf.feats is None or len(kf.feats) == 0:
            return ''
        upto = kf.id - cfg['min_keyframe_gap'] + 1
        if upto <= 0 or kf.id - self.last_loop_kf < cfg['min_keyframes_between_loops']:
            return ''
        self._update_bank(upto)
        if self.bank is None:
            return ''
        votes = vis_mod.vote_candidates(kf.feats.desc, self.bank, self.bank_owner, upto, cfg['vote_similarity'])
        candidates = [int(c) for c in np.argsort(-votes, kind='stable')[:cfg['top_k']] if votes[c] >= cfg['min_votes']]
        if not candidates:
            return 'no_candidate'
        vcfg = dict(self.p['visual'], min_pnp_inliers=cfg['min_pnp_inliers'], min_flow_px=0.0,
                    min_essential_inliers=cfg['min_essential_inliers'], min_length_points=cfg['min_length_points'])
        for c in candidates:
            cand = self.keyframes[c]
            row = dict(query_kf=kf.id, match_kf=c, accepted=0, fitness='', inlier_rmse='', reason='', votes=int(votes[c]),
                       matches=0, essential_inliers=0, pnp_inliers=0, translation_disagreement_m='',
                       rotation_disagreement_deg='', T_rel='')
            self.loop_rows.append(row)
            matches = self.frontend.match(cand.feats, kf.feats)
            v = vis_mod.estimate_relative_pose(cand.feats.kpts, kf.feats.kpts, matches, cand.pts3d, cand.valid3d,
                                               self.ds.K, vcfg, kf.pts3d, kf.valid3d, seed=self.p['seed'] + kf.idx)
            row.update(matches=v['matches'], essential_inliers=v['essential_inliers'], pnp_inliers=v['pnp_inliers'])
            if v['essential_inliers'] < cfg['min_essential_inliers']:
                row['reason'] = 'few_essential_inliers'
                continue
            if not v['ok']:
                row['reason'] = 'pnp_failed'
                continue
            Tv = v['T_ref_cur'].copy()
            Tv[:3, 3] *= cand.scale
            T_init = se3.normalize(cand.T @ Tv)
            w = cfg['submap_half_window']
            try:
                model = self._build_model(range(max(0, c - w), min(c + w + 1, upto)))
            except ValueError:
                row['reason'] = 'candidate_submap_failed'
                continue
            result = self._register(kf.src, kf.src_cov, model, T_init, self.p['registration']['stages'], kf.scale)
            dt, dr = se3.relative_error(T_init, result['T'])
            row.update(fitness=result['fitness'], inlier_rmse=result['inlier_rmse'], translation_disagreement_m=dt,
                       rotation_disagreement_deg=float(np.rad2deg(dr)))
            if result['fitness'] < cfg['min_fitness'] or not result['inlier_rmse'] <= cfg['max_inlier_rmse_m']:
                row['reason'] = 'registration_quality'
                continue
            if dt > cfg['max_translation_disagreement_m'] or np.rad2deg(dr) > cfg['max_rotation_disagreement_deg']:
                row['reason'] = 'visual_geometric_disagreement'
                continue
            T_rel = se3.normalize(se3.inv(cand.T) @ result['T'])
            row.update(accepted=1, reason='accepted', T_rel=' '.join('%.9g' % x for x in T_rel[:3].ravel()))
            self.loops.append((c, kf.id, T_rel))
            self.last_loop_kf = kf.id
            return 'accepted'
        return 'rejected'

    # ------------------------------------------------------------ pose graph
    def optimize(self):
        """以 GTSAM（CPU）最佳化 keyframe pose graph，並把結果寫回 keyframe。

        節點：keyframe pose（SE(3)）。第 0 個 keyframe 加先驗固定規範自由度（gauge）。
        odometry 邊：odometry 鏈上相鄰 keyframe 的相對位姿；閉環邊：通過影像＋點雲驗證的相對位姿，加 Huber 穩健核。
        """
        import gtsam
        g = self.p['graph']
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        sig = lambda r, t: np.array([r, r, r, t, t, t], dtype=np.float64)  # GTSAM Pose3 順序：旋轉在前
        graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(self.odom[self.keyframes[0].idx]),
                                         gtsam.noiseModel.Diagonal.Sigmas(sig(g['prior_sigma'], g['prior_sigma']))))
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(sig(g['odom_rot_sigma_rad'], g['odom_trans_sigma_m']))
        loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(g['huber_k']),
            gtsam.noiseModel.Diagonal.Sigmas(sig(g['loop_rot_sigma_rad'], g['loop_trans_sigma_m'])))
        for kf in self.keyframes:
            initial.insert(kf.id, gtsam.Pose3(se3.normalize(kf.T)))
        for a, b in zip(self.keyframes[:-1], self.keyframes[1:]):
            rel = se3.normalize(se3.inv(self.odom[a.idx]) @ self.odom[b.idx])
            graph.add(gtsam.BetweenFactorPose3(a.id, b.id, gtsam.Pose3(rel), odom_noise))
        for c, q, T_rel in self.loops:
            graph.add(gtsam.BetweenFactorPose3(c, q, gtsam.Pose3(T_rel), loop_noise))
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        for kf in self.keyframes:
            kf.T = se3.normalize(result.atPose3(kf.id).matrix())
        self.optimizations += 1
        return dict(initial_error=float(graph.error(initial)), final_error=float(graph.error(result)))

    def finalize(self):
        """回傳 (odometry poses, optimized poses)。沒有閉環時兩者相同。"""
        odom = [T.copy() for T in self.odom]
        if not self.loops:
            return odom, [T.copy() for T in odom], None
        info = self.optimize()
        optimized = se3.distribute_corrections(odom, [kf.idx for kf in self.keyframes], [kf.T for kf in self.keyframes])
        return odom, optimized, info
