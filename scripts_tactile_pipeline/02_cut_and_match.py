#!/usr/bin/env python3
"""
Merge tactile (left + right) and camera data with absolute times using efficient binary search,
but process multiple videos in parallel.
"""

import cv2
import numpy as np
import os
import sys
import argparse
import datetime
from datetime import timedelta
import re
import subprocess
import json
import pandas as pd

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union
from fractions import Fraction

import av

def iso_or_epoch_to_datetime(time_str: str) -> datetime.datetime:
    """Convert a string to a UTC datetime."""
    if re.match(r'^\d+(\.\d+)?$', time_str.strip()):
        # Epoch time
        epoch_val = float(time_str)
        return datetime.datetime.utcfromtimestamp(epoch_val).replace(tzinfo=datetime.timezone.utc)
    naive_dt = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    return naive_dt.replace(tzinfo=datetime.timezone.utc)

def scan_qr_and_black(video_path, qr_latency, black_threshold, skip_factor=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open {video_path}")
        return None, None, None

    qr_detector = cv2.QRCodeDetector()
    base_timestamp = None
    base_frame_idx = None
    black_idx = None

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Black detection
        if black_idx is None:
            mean_val = gray.mean()
            if mean_val < black_threshold:
                black_idx = frame_idx
                print(f"Found black frame near idx={frame_idx}")

        # 2) QR detection
        if base_timestamp is None:
            data, bbox, _ = qr_detector.detectAndDecode(gray)
            if data:
                try:
                    dt = iso_or_epoch_to_datetime(data)
                    dt += timedelta(seconds=qr_latency)
                    base_timestamp = dt
                    base_frame_idx = frame_idx
                    print(f"Found QR code at ~frame {frame_idx}, base_timestamp={base_timestamp}")
                except Exception as ex:
                    print(f"QR decode/time conv failed at frame {frame_idx}: {ex}")

        if base_timestamp is not None and black_idx is not None:
            break

    cap.release()
    return base_timestamp, base_frame_idx, black_idx

def cut_video_ffmpeg(input_path, output_path, start_time_sec):
    """
    Use ffmpeg to cut 'input_path' at 'start_time_sec', re-encode the video
    so that indexing/metadata is regenerated, saving to 'output_path'.
    """
    if start_time_sec < 0:
        start_time_sec = 0

    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", input_path,
        "-ss", str(start_time_sec),
        "-c:v", "libx264",      # re-encode video to fix indexing
        "-crf", "15",
        "-preset", "slow",
        "-c:a", "copy",         # copy audio without re-encoding
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg cut failed:", e)
        return False
    return True

def load_tactile_data(json_path, topic):
    times = []
    values = []
    print(f"Loading tactile data from {json_path}, topic={topic}")

    with open(json_path, "r") as f:
        data_dict = json.load(f)

    key = f"topic:{topic}"
    if key not in data_dict:
        print(f"ERROR: Tactile data for topic '{topic}' not found in {json_path}")
        return [], []

    sub = data_dict[key]
    times_str = sub["times"]
    values_raw = sub["values"]

    for (t_str, v) in zip(times_str, values_raw):
        try:
            dt = iso_or_epoch_to_datetime(t_str)
            times.append(dt)
            values.append(v)
        except Exception as ex:
            print(f"Failed to parse local_time {t_str}: {ex}")
            continue

    # Sort by ascending datetime
    combined = sorted(zip(times, values), key=lambda x: x[0])
    if not combined:
        print(f"No data found in {json_path} for {topic}.")
        return [], []
    times_sorted, values_sorted = zip(*combined)
    return list(times_sorted), list(values_sorted)

def find_latest_index(tactile_times, target_time):
    i = bisect_right(tactile_times, target_time)
    idx = i - 1
    if idx < 0:
        return None
    if target_time - tactile_times[idx] > timedelta(seconds=5):
        return None
    return idx

def process_cut_video(
    cut_video_path,
    base_timestamp,
    base_frame_index,
    cut_start_index,
    fps,
    left_times,
    left_values,
    right_times,
    right_values,
    npy_save_path,
    plot_video=False,
    plot_images=False
):
    """
    Reads the *cut* video. For each frame i, compute abs_time and find nearest tactile frames.
    Save .npy with shape (num_frames, 12, 64). If either plot_video=True or plot_images=True,
    we create an overlay of camera + tactile, and optionally:
      - if plot_video=True, write that overlay to an MP4
      - if plot_images=True, save each overlay frame to frames/*.png
    When plot_video=True, also save a separate video with only the tactile frames,
    with a small gap between left and right.
    """
    cap = cv2.VideoCapture(cut_video_path)
    if not cap.isOpened():
        print(f"Cannot open cut video {cut_video_path}")
        return 0

    # DEBUG PRINT: check how many frames the cut video has (per cv2)
    # cut_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"DEBUG: The newly cut video '{cut_video_path}' reports {cut_total_frames} frames via cv2.")

    all_tactile = []
    frame_count = 0

    video_writer = None
    tactile_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    do_overlay = plot_video or plot_images
    frames_dir = None
    if plot_images:
        frames_dir = os.path.join(os.path.dirname(npy_save_path), "frames")
        os.makedirs(frames_dir, exist_ok=True)

    # Define gap size in pixels between left and right tactile visuals
    gap_size = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        absolute_time = base_timestamp + timedelta(
            seconds=((cut_start_index + frame_count) - base_frame_index) / fps
        )

        idx_left = find_latest_index(left_times, absolute_time)
        if idx_left is None:
            print(f"No tactile match (left) at frame {frame_count} => {absolute_time}. Aborting video.")
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if tactile_writer is not None:
                tactile_writer.release()
            return 0

        idx_right = find_latest_index(right_times, absolute_time)
        if idx_right is None:
            print(f"No tactile match (right) at frame {frame_count} => {absolute_time}. Aborting video.")
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if tactile_writer is not None:
                tactile_writer.release()
            return 0

        arr_left = np.array(left_values[idx_left], dtype=np.float32).reshape((12, 32))
        arr_right = np.array(right_values[idx_right], dtype=np.float32).reshape((12, 32))
        combined = np.hstack((arr_left, arr_right))  # shape (12,64)
        all_tactile.append(combined)

        if do_overlay:
            left_vis_u8 = (arr_left * 255).astype(np.uint8)
            right_vis_u8 = (arr_right * 255).astype(np.uint8)
            left_color = cv2.applyColorMap(left_vis_u8, cv2.COLORMAP_VIRIDIS)
            right_color = cv2.applyColorMap(right_vis_u8, cv2.COLORMAP_VIRIDIS)

            scale_factor = 10
            left_color_big = cv2.resize(
                left_color,
                (left_color.shape[1]*scale_factor, left_color.shape[0]*scale_factor),
                interpolation=cv2.INTER_NEAREST
            )
            right_color_big = cv2.resize(
                right_color,
                (right_color.shape[1]*scale_factor, right_color.shape[0]*scale_factor),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.putText(left_color_big, "LEFT", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(right_color_big, "RIGHT", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Create a horizontal spacer between left and right tactile visuals
            spacer = np.zeros((left_color_big.shape[0], gap_size, 3), dtype=np.uint8)

            # Stack left, spacer, and right tactile visuals
            tactile_vis = np.hstack((left_color_big, spacer, right_color_big))
            tvis_h, tvis_w = tactile_vis.shape[:2]

            cam_h, cam_w = frame.shape[:2]
            new_cam_h = int(cam_h * (tvis_w / float(cam_w)))
            cam_resized = cv2.resize(frame, (tvis_w, new_cam_h), interpolation=cv2.INTER_AREA)

            final_img = np.vstack((cam_resized, tactile_vis))

            if plot_video:
                if video_writer is None:
                    out_video_path = npy_save_path.replace(".npy", "_overlay.mp4")
                    out_h, out_w = final_img.shape[:2]
                    video_writer = cv2.VideoWriter(
                        out_video_path, fourcc, 60.0, (out_w, out_h), True
                    )
                    out_tactile_path = npy_save_path.replace(".npy", "_tactile.mp4")
                    # Initialize tactile-only writer with tactile_vis dimensions
                    tactile_writer = cv2.VideoWriter(
                        out_tactile_path, fourcc, 60.0, (tvis_w, tvis_h), True
                    )

                # Write overlay frame
                video_writer.write(final_img)
                # Write tactile-only frame
                tactile_writer.write(tactile_vis)

            if plot_images and frames_dir is not None:
                out_frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(out_frame_path, final_img)

    cap.release()
    if frame_count == 0:
        print("No frames found after cut, skipping save.")
        return 0

    all_tactile = np.array(all_tactile, dtype=np.float32)
    np.save(npy_save_path, all_tactile)
    print(f"Saved {frame_count} frames of tactile data => {all_tactile.shape} => {npy_save_path}")

    if video_writer is not None:
        video_writer.release()
        print(f"Overlay video => {npy_save_path.replace('.npy', '_overlay.mp4')}")
    if tactile_writer is not None:
        tactile_writer.release()
        print(f"Tactile-only video => {npy_save_path.replace('.npy', '_tactile.mp4')}")

    # Truncate camera_trajectory.csv to keep last `frame_count` rows
    camera_csv_path = os.path.join(os.path.dirname(cut_video_path), "camera_trajectory.csv")
    if os.path.isfile(camera_csv_path):
        try:
            df = pd.read_csv(camera_csv_path)
            original_csv_len = len(df)  # DEBUG: how many rows total?

            # keep only last frame_count rows
            df = df.tail(frame_count).copy()
            truncated_csv_len = len(df)

            df["frame_idx"] = range(frame_count)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(camera_csv_path, index=False)

            # DEBUG PRINT: how many did we cut vs final # rows
            # print(f"DEBUG: The CSV originally had {original_csv_len} rows; "
            #      f"we truncated to the last {frame_count}, so now it has {truncated_csv_len} rows.")
        except Exception as e:
            print(f"Failed to truncate camera_trajectory.csv: {e}")

    return frame_count

def process_one_video(
    vid,
    final_output_dir,
    left_times, left_values,
    right_times, right_values,
    black_threshold,
    cut_delay,
    qr_latency,
    plot_video,
    plot_images=False
):
    parent_dir = os.path.dirname(vid)
    original_raw_path = os.path.join(parent_dir, "original_raw_video.mp4")
    if os.path.exists(original_raw_path):
        vid = original_raw_path
        print("Found original_raw_video.mp4, skipping rename.")
    else:
        try:
            os.rename(vid, original_raw_path)
            vid = original_raw_path
            print("Renamed raw_video.mp4 => original_raw_video.mp4")
        except Exception as e:
            print(f"Cannot rename {vid} => {original_raw_path}, error: {e}")
            return False

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"Cannot open {vid}, skipping.")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {vid} => FPS={fps:.3f}, total frames={total_frames}")

    base_ts, base_idx, black_idx = scan_qr_and_black(vid, qr_latency, black_threshold)
    if base_ts is None or base_idx is None:
        print("No valid QR code found => skipping.")
        return False
    if black_idx is None:
        print("No black frame found => skipping.")
        return False

    cut_time = (black_idx / fps) + cut_delay
    cut_start_index = int(round(cut_time * fps))
    print(f"Cut time = {cut_time:.2f}s => frame idx ~{cut_start_index}")

    out_cut_mp4 = os.path.join(final_output_dir, "raw_video.mp4")
    success = cut_video_ffmpeg(vid, out_cut_mp4, cut_time)
    if not success:
        print("Cut failed => skipping.")
        return False

    # debug_cap = cv2.VideoCapture(out_cut_mp4)
    # debug_cut_frames = int(debug_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # debug_cap.release()
    # print(f"DEBUG: Just cut '{out_cut_mp4}', it has {debug_cut_frames} frames per cv2.")

    out_npy = os.path.join(final_output_dir, "tactile.npy")
    final_count = process_cut_video(
        cut_video_path=out_cut_mp4,
        base_timestamp=base_ts,
        base_frame_index=base_idx,
        cut_start_index=cut_start_index,
        fps=fps,
        left_times=left_times,
        left_values=left_values,
        right_times=right_times,
        right_values=right_values,
        npy_save_path=out_npy,
        plot_video=plot_video,
        plot_images=plot_images
    )
    print(f"Final frame count after cut = {final_count}")
    if final_count == 0:
        return False

    imu_json_path = os.path.join(parent_dir, "imu_data.json")
    if os.path.isfile(imu_json_path):
        try:
            with open(imu_json_path, "r") as f:
                imu_data = json.load(f)
            cut_samples = int(cut_time * 200)
            streams_obj = imu_data["1"]["streams"]
            for stype in ["ACCL", "GYRO"]:
                if stype in streams_obj:
                    old_samples = streams_obj[stype]["samples"]
                    new_samples = old_samples[cut_samples:]
                    for s in new_samples:
                        s["cts"] -= cut_time
                    streams_obj[stype]["samples"] = new_samples
                    print(f"IMU {stype} => trimmed from {len(old_samples)} to {len(new_samples)}.")
            with open(imu_json_path, "w") as f:
                json.dump(imu_data, f)
            print(f"IMU trimmed => {imu_json_path}")
        except Exception as e:
            print(f"Failed to trim IMU: {e}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Merge left+right tactile (12×32 each => 12×64) with camera frames. Optimized single-pass scanning."
    )
    parser.add_argument("--video", nargs='+', required=True,
                        help="One or more .MP4 files (or directories) to process.")
    parser.add_argument("--bag", required=True,
                        help="Path to the .json file containing tactile data.")
    parser.add_argument("--tactile_topic_left", default="/tactile_input_left",
                        help="Left Tactile topic.")
    parser.add_argument("--tactile_topic_right", default="/tactile_input_right",
                        help="Right Tactile topic.")
    parser.add_argument("--black_threshold", type=float, default=10.0,
                        help="Threshold for black frame detection.")
    parser.add_argument("--cut_delay", type=float, default=1.5,
                        help="Seconds after black frame to cut.")
    parser.add_argument("--qr_latency", type=float, default=0.12,
                        help="Latency offset for the QR code's parsed time.")
    parser.add_argument("--plot_video", action="store_true",
                        help="If set, also create a side-by-side overlay video.")
    parser.add_argument("--plot_images", action="store_true",
                        help="If set, save per-frame overlay images under 'frames/'.")
    args = parser.parse_args()

    video_list = []
    for item in args.video:
        if os.path.isdir(item):
            mp4s = sorted([
                os.path.join(item, f)
                for f in os.listdir(item)
                if f.lower().endswith(".mp4")
            ])
            video_list.extend(mp4s)
        elif os.path.isfile(item):
            video_list.append(item)
        else:
            print(f"WARNING: {item} is not a file or directory, skipping.")

    if not video_list:
        print("No .mp4 files found.")
        sys.exit(1)

    left_times, left_values = load_tactile_data(args.bag, args.tactile_topic_left)
    right_times, right_values = load_tactile_data(args.bag, args.tactile_topic_right)
    if (not left_times) and (not right_times):
        print("No tactile messages for either topic => exit.")
        sys.exit(1)

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for vid in video_list:
            futures.append(
                executor.submit(
                    process_one_video,
                    vid,
                    os.path.dirname(vid),
                    left_times, left_values,
                    right_times, right_values,
                    args.black_threshold,
                    args.cut_delay,
                    args.qr_latency,
                    args.plot_video,
                    args.plot_images
                )
            )

        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Video processing exception: {e}")
                fail_count += 1

    print("All videos processed.")
    print(f"Successful videos: {success_count}, Failed videos: {fail_count}")

if __name__ == "__main__":
    main()
