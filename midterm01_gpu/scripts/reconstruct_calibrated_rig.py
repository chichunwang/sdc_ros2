#!/usr/bin/env python3
"""少量原始影像 MapAnything 重建；主機獨立 ML venv，不更動 ROS Python。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from calibrated_rig_geometry import (OPTICAL_TO_FLU, PriorTrajectory, load_rig, rectify,
                                     backproject, transform, voxel_indices, balanced_indices, write_ply)
from three_camera_source import sha256, camera_headers, resolve_camera_timestamps


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(args):
    root = Path(__file__).resolve().parents[1]
    args.output.resolve().relative_to((root / 'outputs').resolve())
    groups = [tuple(map(int, part.split(':'))) for part in args.groups.split(',')]
    if not 1 <= len(groups) <= 24 or len(set(groups)) != len(groups) or any(t not in (1, 2) or i < 1 for t, i in groups):
        raise ValueError('Choose 1–24 unique Track:index groups away from the first time boundary')
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__, 'opencv': cv2.__version__,
                'device': torch.cuda.get_device_name(), 'checkpoint_sha256': sha256(args.checkpoint / 'model.safetensors'),
                'checkpoint_config_sha256': sha256(args.checkpoint / 'config.json'),
                'upstream_commit': subprocess.check_output(['git', '-C', str(args.model_repo), 'rev-parse', 'HEAD'], text=True).strip(),
                'checkpoint_id': 'facebook/map-anything-apache', 'checkpoint_revision': args.checkpoint_revision,
                'resolution_wh': [args.width, args.height], 'command': sys.argv,
                'autocast_weight_cache': False,
                'scope': 'Small diagnostic, no ATE/Kaggle; metric depth is a learned prediction, not stereo ground truth'}
    dump(args.output / 'runtime.json', versions)
    (args.output / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
                                             if shutil.which('pip', path=str(Path(sys.executable).parent)) else 'See uv_freeze.txt in parent experiment\n')
    snapshots = args.output / 'sources'
    snapshots.mkdir()
    for name in ('reconstruct_calibrated_rig.py', 'calibrated_rig_geometry.py', 'three_camera_source.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshots / name)
    start = time.perf_counter()
    model = MapAnything.from_pretrained(str(args.checkpoint)).eval().to('cuda')
    versions['model_load_seconds'] = time.perf_counter() - start
    reports, trajectories, indices, headers, rigs = {}, {}, {}, {}, {}
    for track in sorted({t for t, _ in groups}):
        prior_run = getattr(args, f'prior_track{track}')
        metadata = json.loads((prior_run / 'validation.json').read_text())
        if metadata['track'] != track or metadata['sensor_fusion'] != 'heading_rts':
            raise ValueError('Wrong original prior Track/fusion')
        trajectories[track] = PriorTrajectory(prior_run / 'sensor_prior.csv')
        indices[track] = list(csv.DictReader((args.data_root / f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv').open()))
        raw_headers = camera_headers(args.data_root / f'Track{track}/Track{track}.db3')
        headers[track] = {tag: resolve_camera_timestamps(raw_headers[tag], [int(r[f'{tag}_ts']) for r in indices[track]])
                          for tag in ('f', 'fl', 'fr')}
        directory = args.data_root / f'Track{track}/Track{track}_images/camera_info'
        rigs[track] = load_rig(directory)
        dump(args.output / f'rig_track{track}.json', {
            'fixed_optical_front_from_camera': {k: v.tolist() for k, v in rigs[track].items()},
            'relative_height_flu_m': {k: float((OPTICAL_TO_FLU @ v[:3, 3])[2]) for k, v in rigs[track].items()},
            'calibration_sha256': {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
            'prior_sha256': sha256(prior_run / 'sensor_prior.csv'),
            'motion_contract': 'Assume sensor prior rotation/translation represents front-camera FLU motion; body lever arm unverified',
            'output_reference': 'front BAG timestamp; capture candidates use exact ROS headers; no camera height estimation'})
    for track, original_idx in groups:
        row = indices[track][original_idx]
        stamp = int(row['f_ts'])
        group_path = args.output / f'track{track}_{original_idx:06d}'
        group_path.mkdir()
        trajectory, rig = trajectories[track], rigs[track]
        image_root = args.data_root / f'Track{track}/Track{track}_images'
        reference_header = headers[track]['f'][original_idx][1]
        views, validity, source_images, capture_poses = [], [], [], []
        for tag in ('f', 'fl', 'fr'):
            image = image_root / f'Track{track}_{tag}_images/{original_idx:06d}_ts_{row[tag + "_ts"]}.jpg'
            calibration = image_root / f'camera_info/gige_100_{tag}_hdr_camera_info.yaml'
            mask = image_root / 'camera_info/gige_100_f_hdr_mask.png' if tag == 'f' else None
            rgb, k, valid = rectify(image, calibration, (args.width, args.height), mask)
            capture_stamp = headers[track][tag][original_idx][1]
            pose = trajectory.camera_pose(reference_header, capture_stamp, rig[tag])
            capture_poses.append(pose)
            views.append({'img': rgb, 'intrinsics': k, 'camera_poses': pose.astype(np.float32), 'is_metric_scale': True})
            validity.append(valid)
            source_images.append({'camera': tag, 'filename': str(image), 'sha256': sha256(image),
                                  'bag_ns': headers[track][tag][original_idx][0], 'header_ns': capture_stamp,
                                  'index_time_basis': headers[track][tag][original_idx][3]})
            cv2.imwrite(str(group_path / f'{tag}_rectified.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(group_path / f'{tag}_valid.png'), valid.astype('uint8') * 255)
        group_summary = {'track': track, 'original_idx': original_idx, 'timestamp_ns': stamp, 'images': source_images, 'arms': {}}
        for arm in ('single', 'predicted', 'rig'):
            # All arms use the same rectification, checkpoint, precision and resolution.
            current = [dict(v) for v in (views[:1] if arm == 'single' else views)]
            if arm != 'rig':
                for v in current:
                    v.pop('camera_poses')
            inputs = preprocess_inputs(current, resize_mode='fixed_size', size=(args.width, args.height))
            for a, b in zip(inputs, current):
                if tuple(a['img'].shape[-2:]) != (args.height, args.width):
                    raise ValueError('Unexpected model resize; validity mask no longer aligned')
                np.testing.assert_allclose(a['intrinsics'][0].numpy(), b['intrinsics'], atol=1e-4)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before = time.perf_counter()
            # Let infer enable AMP only for its forward pass. Postprocessing
            # includes float32 linear solves unsupported for bfloat16.
            with torch.inference_mode(), torch.autocast('cuda', enabled=False, cache_enabled=False):
                predictions = model.infer(inputs, memory_efficient_inference=True, minibatch_size=1,
                                          use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True,
                                          apply_confidence_mask=True, confidence_percentile=10)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            peak = torch.cuda.max_memory_allocated()
            points, colors, ids, confidences, camera_report = [], [], [], [], []
            predicted_reference = None
            for camera_id, (pred, v) in enumerate(zip(predictions, views)):
                def array(key):
                    return pred[key].detach().float().cpu().numpy()[0]
                depth, confidence, model_mask = array('depth_z')[..., 0], array('conf'), array('mask')[..., 0].astype(bool)
                model_pose = array('camera_poses')
                if predicted_reference is None:
                    predicted_reference = model_pose.copy()
                # Enforce calibrated rays and fixed rig poses explicitly: pose inputs alone are neural conditioning.
                cam_points = backproject(depth, v['intrinsics'])
                merge_pose = capture_poses[camera_id] if arm == 'rig' else np.linalg.inv(predicted_reference) @ model_pose
                header_points = transform(cam_points, merge_pose)
                header_flu = header_points @ OPTICAL_TO_FLU.T
                bag_from_header = np.linalg.inv(trajectory.pose(stamp)) @ trajectory.pose(reference_header)
                merged = transform(header_flu, bag_from_header)
                ranges = np.linalg.norm(cam_points, axis=-1)
                keep = validity[camera_id] & model_mask & np.isfinite(depth) & (depth > 0) & (ranges >= 2.) & (ranges <= 60.)
                keep &= np.all(np.isfinite(merged), axis=-1) & np.isfinite(confidence)
                rgb = v['img']
                np.savez_compressed(group_path / f'{arm}_{camera_id}.npz', depth_z=depth, confidence=confidence,
                                    mask=keep, intrinsics=v['intrinsics'], model_camera_pose=model_pose,
                                    applied_camera_pose=merge_pose, bag_from_header_flu=bag_from_header,
                                    camera_id=np.uint8(camera_id), header_ns=np.int64(source_images[camera_id]['header_ns']))
                points.append(merged[keep]); colors.append(rgb[keep]); ids.append(np.full(keep.sum(), camera_id, dtype=np.uint8)); confidences.append(confidence[keep])
                camera_report.append({'camera': source_images[camera_id]['camera'], 'kept': int(keep.sum()),
                                      'predicted_translation_m': model_pose[:3, 3].tolist(),
                                      'applied_translation_m': merge_pose[:3, 3].tolist(),
                                      'metric_scaling_factor': float(array('metric_scaling_factor').ravel()[0])})
            points, colors, ids, confidences = map(np.concatenate, (points, colors, ids, confidences))
            selections = {}
            uniform = voxel_indices(points)
            if len(uniform) > args.budget:
                uniform = uniform[np.linspace(0, len(uniform) - 1, args.budget, dtype=int)]
            selections[arm] = (uniform, None)
            if arm == 'rig':
                front = np.flatnonzero(ids == 0)
                front = front[voxel_indices(points[front])]
                if len(front) > args.budget:
                    front = front[np.linspace(0, len(front) - 1, args.budget, dtype=int)]
                selections['rig_front'] = (front, None)
                selections['balanced'] = balanced_indices(points, ids, confidences, args.budget)
                matched = voxel_indices(points)
                matched_count = min(len(matched), len(selections['balanced'][0]))
                if matched_count < 10:
                    raise ValueError('Insufficient balanced geometry for a controlled comparison')
                matched = matched[np.linspace(0, len(matched) - 1, matched_count, dtype=int)]
                selections['rig_uniform_matched'] = (matched, {'matched_to': 'balanced', 'points': matched_count})
                np.savez_compressed(group_path / 'rig_candidates.npz', points=points, rgb=colors, camera_ids=ids, confidence=confidences)
                # Same inferred depth, suppress motion compensation only: isolates synchronization effect.
                static = []
                for camera_id in range(3):
                    saved = np.load(group_path / f'rig_{camera_id}.npz')
                    cp = backproject(saved['depth_z'], saved['intrinsics'])
                    pp = transform(cp, rig[('f', 'fl', 'fr')[camera_id]]) @ OPTICAL_TO_FLU.T
                    static.append(transform(pp, bag_from_header)[saved['mask']])
                static_points = np.concatenate(static)
                selections['rig_static'] = (uniform, {'motion_compensation_applied': False})
            for variant, (selection, balance_report) in selections.items():
                directory = args.output / f'{variant}_track{track}'
                directory.mkdir(exist_ok=True)
                filename = f'{original_idx:06d}_ts_{stamp}.ply'
                selected_points = static_points if variant == 'rig_static' else points
                write_ply(directory / filename, selected_points[selection], colors[selection])
                np.savez_compressed(directory / f'{original_idx:06d}_labels.npz', camera_ids=ids[selection], confidence=confidences[selection])
                key = (variant, track)
                reports.setdefault(key, {'schema': 'calibrated_rig_v1', 'track': track, 'variant': variant,
                                         'coordinate_frame': 'front_camera_FLU', 'reference_time': 'front_camera_bag_timestamp',
                                         'ply_scale': 1., 'experimental': True, 'frames': [],
                                         'fixed_rig_heights': variant.startswith('rig') or variant == 'balanced',
                                         'per_camera_motion_compensation': variant in ('rig', 'rig_front', 'balanced', 'rig_uniform_matched'),
                                         'budget': args.budget, 'metric_status': 'Learned metric depth conditioned on intrinsics/rig; no fitted scale',
                                         'runtime_sha256': sha256(args.output / 'runtime.json')})
                reports[key]['frames'].append({'original_idx': original_idx, 'timestamp_ns': stamp, 'filename': filename,
                                               'sha256': sha256(directory / filename), 'points': len(selection),
                                               'camera_counts': {str(i): int(np.sum(ids[selection] == i)) for i in range(3)},
                                               'sampling': balance_report})
            group_summary['arms'][arm] = {'seconds': seconds, 'peak_cuda_bytes': peak, 'camera_report': camera_report}
            del predictions, inputs
            torch.cuda.empty_cache()
            print(json.dumps({'track': track, 'idx': original_idx, 'arm': arm, 'seconds': round(seconds, 3), 'peak_gb': peak / 1e9}), flush=True)
        dump(group_path / 'report.json', group_summary)
    for (variant, track), report in reports.items():
        report['frames'].sort(key=lambda r: r['original_idx'])
        dump(args.output / f'{variant}_track{track}/report.json', report)
    versions['total_seconds'] = time.perf_counter() - start
    dump(args.output / 'completion.json', versions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-revision', default='00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a')
    p.add_argument('--model-repo', type=Path, required=True)
    p.add_argument('--prior-track1', type=Path, required=True)
    p.add_argument('--prior-track2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', default='1:1505,1:1506,1:1823,1:1824,2:1661,2:1662,2:1397,2:1398')
    p.add_argument('--width', type=int, default=448)
    p.add_argument('--height', type=int, default=294)
    p.add_argument('--budget', type=int, default=6000)
    args = p.parse_args()
    if args.width < 112 or args.height < 112 or args.width % 14 or args.height % 14 or args.budget < 100:
        p.error('Resolution must be multiples of 14, >=112; budget >=100')
    run(args)


if __name__ == '__main__':
    main()
