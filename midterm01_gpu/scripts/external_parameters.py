"""作者 CPU 流程的跨 Track 設定：直接在本檔修改，不改原始作者程式。"""
from copy import deepcopy

PROCESSES = 2
# H=1.573 與 nominal=8.6 沿用作者假設，並非本次標定值。
TRACKS = {
    1: dict(data_relative="midterm01/data/Track1", expected_frames=3011,
            init_search=False, gyro_dt_s=0.0, map_reader="author_ascii"),
    2: dict(data_relative="midterm01/data/Track2", expected_frames=3299,
            init_search=False, gyro_dt_s=0.0, map_reader="binary_xyz_adapter"),
    3: dict(data_relative="midterm01/data/Track3", expected_frames=3142,
            init_search=True, gyro_dt_s=0.14, map_reader="author_ascii"),
}


def get_parameters(track=3):
    if track not in TRACKS:
        raise ValueError("Track 必須是 1、2 或 3")
    value = deepcopy(TRACKS[track])
    value.update(track=track, bag_name=f"Track{track}.db3", processes=PROCESSES,
                 result_name=f"result_{track}.csv", author_output_name=f"track{track}_cam_causal.csv",
                 height_m=1.573, nominal_scale=8.6, ranges_m=[12, 15, 18, 25],
                 min_xy_info=2.0, backfill=0, author_algorithm_modified=False,
                 gnss_used_for_estimation=False, ply_subdir="raw_ply",
                 height_status="author transferred fixed-height assumption; not calibrated here",
                 gyro_dt_status="author Track3 override" if track == 3 else "author CLI default; not clock calibration",
                 io_compatibility_fix="unbuffered input in ply.py and convert_map.py",
                 map_time_fields_used=False)
    if not isinstance(value["processes"], int) or value["processes"] < 1:
        raise ValueError("PROCESSES 必須是正整數")
    if value["backfill"] != 0:
        raise ValueError("研究入口不開啟 backward backfill")
    return value
