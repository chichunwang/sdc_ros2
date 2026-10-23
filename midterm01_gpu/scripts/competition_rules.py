"""因果定位規則與已轉貼的資料契約；公告確認不等於本機資料或 runner 已驗證。"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES_VERSION = '2026-10-22'
RULES_REVIEW_DATE = '2026-10-23'
DOCUMENTS = {
    'original': ('00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf',
                 'cdf6467c50562a03e3a34d4ab26edb4db8ace84da9ca22a726728c0ada8ea047'),
    'updates': ('01_Mid01_Updates_Rules.pdf',
                'cf60236b87f6a89b12f1b182dfd27b20a356023636877677ef0b608c2e8b14fd'),
}
TRACKS = {
    1: {'ranking_weight': .3, 'expected_frames': 3011, 'gnss_pose_provided': True},
    2: {'ranking_weight': .3, 'expected_frames': 3299, 'gnss_pose_provided': True},
    3: {'ranking_weight': .4, 'expected_frames': 3142, 'gnss_pose_provided': False,
        'public_prefix_percent_range': [5, 40], 'private_baseline': 1.40,
        'private_baseline_comparison': '<=',
        'dataset_description_source': {
            'kind': 'user_provided_Kaggle_Dataset_Description_Topics_Submission',
            'recorded_date': '2026-10-23',
            'url': 'https://www.kaggle.com/competitions/sdc-2026-localization-track-03/data',
            'webpage_independently_verified': False,
        },
        'announced_sensor_topics': [
            '/lucid_cameras_x00/gige_100_f_hdr/h265',
            '/lucid_cameras_x01/gige_100_fl_hdr/h265',
            '/lucid_cameras_x01/gige_100_fr_hdr/h265',
            '/imu/data',
        ],
        'imu_announced_measurements': ['orientation', 'acceleration']},
}
RANKING_BASE_POINTS = {
    'first': 56, 'second': 52, 'third': 48, 'top_50_percent': 42,
    'top_75_percent': 36, 'others_above_baseline': 30, 'late': 22, 'below_baseline': 0,
}


def official_pdf_path(kind='original'):
    """單一正式原件位置；不重新建立已搬走的 CPU docs PDF。"""
    return ROOT / 'doc' / 'Competition Regulations' / DOCUMENTS[kind][0]


def document_identities():
    result = {}
    for kind, (_, expected) in DOCUMENTS.items():
        path = official_pdf_path(kind)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result[kind] = {'path': str(path), 'sha256': actual,
                        'expected_sha256': expected, 'matches_reviewed_source': actual == expected}
    return result


def track_contract(track):
    if track not in TRACKS:
        raise ValueError('Track must be 1, 2 or 3')
    return dict(TRACKS[track], track=track,
                frame_count_confirmed=True,
                submission_filename=f'result_{track}.csv',
                submission_header=['idx', 'x', 'y', 'yaw'],
                submission_format_confirmed=True,
                submission_format_scope='CSV columns and data-row count only',
                expected_submission_rows=TRACKS[track]['expected_frames'],
                expected_csv_records_including_header=TRACKS[track]['expected_frames']+1,
                submission_sample_verified=False,
                submission_contract_confirmed=False,
                gnss_input_allowed=track in (1, 2),
                historical_gnss_smoother_available=track in (1, 2),
                gnss_smoother_supported=False,
                causal_localization_required=True,
                backward_smoothing_allowed=False,
                offline_trajectory_correction_allowed=False,
                causal_pipeline_verified=False,
                submission_compliance_verified=False,
                submission_ready=False,
                map_localization_fields=['x', 'y', 'z'],
                map_timestamp_reference_used=False,
                leaderboard_error_inference_allowed=False)
