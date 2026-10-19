#!/usr/bin/env python3
"""用獨立合成資料驗證 CUDA pose 方向、數千公尺原點及無對應點回退；不代表實際精度。"""

import csv
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent.parent
    from accuracy.benchmark import create_run_directory
    run = create_run_directory(root / "outputs/run", "cuda_contract")
    container_root = "/root/outputs/run/" + run.name
    rng = random.Random(20261017)
    source = [(cluster % 5 * 2.7 + rng.gauss(0, .2), (cluster // 5) % 4 * 3.1 + rng.gauss(0, .2),
               cluster // 20 * 2.2 + rng.gauss(0, .2)) for cluster in range(60) for _ in range(120)]
    angle = .1
    c, s = math.cos(angle), math.sin(angle)
    translation = [-4799.5, 2279.7, 104.2]
    guess = [[c, -s, 0, translation[0] + .1], [s, c, 0, translation[1] - .08],
             [0, 0, 1, translation[2]], [0, 0, 0, 1]]
    target = [(c * x - s * y + translation[0], s * x + c * y + translation[1], z + translation[2]) for x, y, z in source]
    (run / "ply").mkdir()
    with (run / "ply/000000_ts_1000000000.ply").open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(source)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        stream.writelines(f"{x:.8f} {y:.8f} {z:.8f} 100 100 100\n" for x, y, z in source)
    (run / "init_guess.txt").write_text("\n".join(" ".join(map(str, row)) for row in guess) + "\n")
    (run / "truth.json").write_text(json.dumps({"translation": translation, "yaw_rad": angle, "seed": 20261017}, indent=2) + "\n")
    checks = []
    for mode, no_overlap, use_defaults in [("P2D", False, False), ("D2D", False, False),
                                          ("P2D", True, False), ("D2D", True, False), ("D2D", False, True)]:
        name = ("default" if use_defaults else mode.lower()) + ("_no_overlap" if no_overlap else "_global_origin")
        case = run / name
        case.mkdir()
        with (case / "map.pcd").open("w") as stream:
            stream.write(f"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(target)}\nHEIGHT 1\nPOINTS {len(target)}\nDATA ascii\n")
            stream.writelines(f"{x:.8f} {y + (60 if no_overlap else 0):.8f} {z:.8f}\n" for x, y, z in target)
        params = {
            "map_path": container_root + "/" + name, "ply_path": container_root + "/ply",
            "init_guess_file": container_root + "/init_guess.txt", "save_path": container_root + "/" + name,
            "save_file": "result.txt", "registration_backend": "cuda_ndt", "cuda_ndt_mode": mode,
            "ply_scale": 1.0, "map_voxel_leaf_size": 0.0, "input_voxel_leaf_size": 0.0,
            "ndt_resolution": 1.0, "cuda_translation_epsilon": .001, "cuda_rotation_epsilon": .001,
            "ndt_max_iterations": 60, "max_frames": 1, "keep_alive": False,
        }
        if use_defaults:
            del params["registration_backend"]
            del params["cuda_ndt_mode"]
        (case / "params.yaml").write_text(json.dumps({"image_point_cloud_localizer": {"ros__parameters": params}}, indent=2) + "\n")
        command = ["docker", "exec", "sdc_midterm01_gpu", "bash", "-c",
                   "source /root/midterm/install/setup.bash && ros2 run localization localization --ros-args --params-file " + container_root + "/" + name + "/params.yaml"]
        if use_defaults:
            launch_args = {("data_path" if key == "map_path" else key): value for key, value in params.items()}
            command[-1] = "source /root/midterm/install/setup.bash && ros2 launch localization localization.launch.py " + shlex.join(
                f"{key}:={str(value).lower() if isinstance(value, bool) else value}" for key, value in launch_args.items())
        result = subprocess.run(command, capture_output=True, text=True)
        (case / "run.log").write_text(result.stdout + result.stderr)
        pose = [float(value) for value in (case / "result.txt").read_text().split()]
        with (case / "timings.csv").open() as stream:
            timing = next(csv.DictReader(stream))
        expected = [guess[i][3] for i in range(3)] if no_overlap else translation
        error = math.sqrt(sum((pose[i * 4 + 3] - expected[i]) ** 2 for i in range(3)))
        rotation_error = abs(math.remainder(math.atan2(pose[4], pose[0]) - angle, 2 * math.pi))
        passed = (result.returncode == 0 and len(pose) == 12 and all(map(math.isfinite, pose))
                  and ((timing["converged"] == "0" and error < .001) if no_overlap
                       else (timing["converged"] == "1" and error < .15 and rotation_error < .03)))
        if use_defaults:
            passed = passed and "Registration backend: cuda_ndt D2D/DIRECT7" in (result.stdout + result.stderr)
        checks.append({"case": name, "passed": passed, "translation_error_m": error,
                       "yaw_error_rad": rotation_error, "converged": timing["converged"]})
    report = {"scope": "Synthetic transform/large-origin/no-overlap contract only; not real-data accuracy.", "checks": checks}
    (run / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(run)
    print(json.dumps(report, indent=2))
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
