import csv
import math
import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def rotation_matrix_to_yaw(pose):
    r11 = pose[0]
    r21 = pose[4]
    return math.atan2(r21, r11)

def txt_to_csv(input_txt, output_csv):
    with open(input_txt, 'r') as txt_file, open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['idx', 'x', 'y', 'yaw'])

        for idx, line in enumerate(txt_file):
            pose = list(map(float, line.split()))
            x = pose[3]
            y = pose[7]
            yaw = rotation_matrix_to_yaw(pose)
            writer.writerow([idx, x, y, yaw])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pose TXT to CSV")
    parser.add_argument("input", help="Source TXT file path")
    parser.add_argument("output", help="Target CSV file path")
    
    args = parser.parse_args()
    txt_to_csv(args.input, args.output)

# python3 txt_to_idx_x_y_yaw.py /root/midterm/src/localization/result/track1_result.txt /root/midterm/src/localization/result/kaggle_track1_result.csv
# python3 txt_to_idx_x_y_yaw.py /root/midterm/src/localization/result/track2_result.txt /root/midterm/src/localization/result/kaggle_track2_result.csv