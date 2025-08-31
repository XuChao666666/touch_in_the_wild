#!/usr/bin/env python3
"""
Merge tactile (left + right) and camera data with absolute times using efficient binary search,
but process multiple videos in parallel.
整段代码实现的功能是：实现了一个 多模态数据（视觉+触觉）时间对齐与融合处理系统，核心功能是通过视频中的时间标记（QR码）和触觉传感器数据，生成时间严格同步的视觉-触觉数据集。
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

# 将两种不同格式的时间字符串转换为带时区信息的UTC datetime 对象。
def iso_or_epoch_to_datetime(time_str: str) -> datetime.datetime:
    """Convert a string to a UTC datetime."""
    # 检查是否是Unix 时间戳
    if re.match(r'^\d+(\.\d+)?$', time_str.strip()):    # 去除收尾的空白之后，判断是否是可以合法的数字
        # Epoch time
        epoch_val = float(time_str)                     # 将字符串处理为浮点数
        return datetime.datetime.utcfromtimestamp(epoch_val).replace(tzinfo=datetime.timezone.utc)  # 将Epoch 时间（存在某一时间起点）转换为UTC时间
    naive_dt = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")    # 使用strptime 解析ISO格式字符串，生成的 naive_dt 是无时区信息的对象，需手动添加UTC时区。
    return naive_dt.replace(tzinfo=datetime.timezone.utc)   # 手动添加操作，说明“这个时间对象应该被理解为UTC时间”。

"""
    输入参数：
        video_path：输入视频文件路径
        qr_latency：QR码时间戳的补偿值（秒），用于调整时间同步。
        black_threshold：判定“黑帧”的灰度均值阈值。
        skip_factor（可选）：帧跳过因子（未在函数内使用，可能为预留参数）。
    返回值:
        base_timestamp：QR码解析的基准时间（UTC datetime）。
        base_frame_idx：QR码出现的帧索引。
        black_idx：检测到黑帧的帧索引。
    黑帧：
        定义：黑帧（Black Frame） 是指视频中所有像素（或绝大部分像素）的RGB值接近（0, 0, 0）的纯黑色画面。
        特征：
            灰度均值极低：通过计算帧的灰度图像均值（gray.mean() ），若低于设定的阈值（如 black_threshold=10），则判定为黑帧。
            全黑或几乎不可见内容，常见于视频的起始/结束、设备开关机瞬间或人工插入的标记帧。
"""
def scan_qr_and_black(video_path, qr_latency, black_threshold, skip_factor=1):
    # 视频初始化，使用opencv打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open {video_path}")
        return None, None, None

    qr_detector = cv2.QRCodeDetector()  # OpenCV的QR码检测器，用于识别和解码QR码。
    base_timestamp = None               # QR码解析的时间戳
    base_frame_idx = None               # QR码出现的帧号 
    black_idx = None                    # 黑帧出现的帧号 

    frame_idx = -1                      # 当前帧索引（从0开始计数）
    while True:
        ret, frame = cap.read()         # 读取下一帧，ret表示是否成功；frame 是读取出来的帧；
        if not ret:
            break
        frame_idx += 1                  # 帧索引+1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将BGR帧转换为灰度图像，供后续QR码检测和黑帧分析使用。

        # 1) Black detection  可能用于标记视频结束或场景切换。
        if black_idx is None:           # 黑帧索引为None
            mean_val = gray.mean()      # 计算灰度图像均值 mean_val，若低于阈值则判定为黑帧。
            if mean_val < black_threshold: # 灰度图像的均值小于黑度阈值
                black_idx = frame_idx   # 获取黑帧
                print(f"Found black frame near idx={frame_idx}")

        # 2) QR detection
        if base_timestamp is None:      # 如果QR码解析的基准时间为None
            data, bbox, _ = qr_detector.detectAndDecode(gray)   # 使用 detectAndDecode 检测QR码，提取内容 data。
            if data:                    # 如果data存在
                try:
                    dt = iso_or_epoch_to_datetime(data) # 调用 iso_or_epoch_to_datetime 将 data 转换为UTC时间（支持Epoch或ISO格式）。
                    dt += timedelta(seconds=qr_latency) # 添加时间补偿 qr_latency（校准硬件延迟）。
                    base_timestamp = dt                 # 更新base_timestamp
                    base_frame_idx = frame_idx          # 将base_frame_idx设置为frame_idx
                    print(f"Found QR code at ~frame {frame_idx}, base_timestamp={base_timestamp}")  
                except Exception as ex:
                    print(f"QR decode/time conv failed at frame {frame_idx}: {ex}") # 容错：捕获时间解析异常，打印错误但不中断流程。

        if base_timestamp is not None and black_idx is not None:    # 当QR码和黑帧均被检测到时，提前退出循环以节省计算资源。
            break

    cap.release()   # 释放视频资源，返回检测结果。未找到的目标保持为 None。
    return base_timestamp, base_frame_idx, black_idx

"""
    输入参数：
        input_path：输入视频文件路径。
        output_path：剪切后视频的输出路径。
        start_time_sec：剪切起始时间（秒），从该时间点开始保留后续内容。
    返回参数：
        True：剪切成功。
        False：剪切失败（打印错误信息）。
"""
def cut_video_ffmpeg(input_path, output_path, start_time_sec):
    """
    Use ffmpeg to cut 'input_path' at 'start_time_sec', re-encode the video
    so that indexing/metadata is regenerated, saving to 'output_path'.

    使用ffmpeg在'start_time_sec'处剪切'input_path'，重新编码视频因此索引/元数据被重新生成，保存到“output_path”。
    """
    if start_time_sec < 0:      # 确保切入时间非负
        start_time_sec = 0

    # 构建FFmpeg命令
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",   # 仅输出错误日志（减少控制台噪音）
        "-i", input_path,                       # 输入文件路径 
        "-ss", str(start_time_sec),             # 剪切起始时间（秒）
        "-c:v", "libx264",      # re-encode video to fix indexing  视频编码器：H.264
        "-crf", "15",           # 视频质量（15为高质量，范围0-51）
        "-preset", "slow",      # 编码速度与压缩效率的权衡（慢速=更高压缩率）
        "-c:a", "copy",         # copy audio without re-encoding
        output_path             # 输出文件路径
    ]
    """
        通过 subprocess.run 调用FFmpeg命令，check=True 表示若命令返回非零状态码则抛出异常。
        捕获异常并打印错误信息（如输入文件不存在、编码器不支持等）。
        返回执行状态（成功/失败）。
    """
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg cut failed:", e)
        return False
    return True

"""
    输入参数：
        json_path：包含触觉数据的JSON文件路径。
        topic：需要提取的数据主题（如传感器名称或类型）。
    输出参数：
        times：解析后的时间戳列表（datetime 对象）。
        values：对应的触觉数值列表。
"""
def load_tactile_data(json_path, topic):
    times = []
    values = []
    print(f"Loading tactile data from {json_path}, topic={topic}")  # 打印加载信息

    with open(json_path, "r") as f:
        data_dict = json.load(f)    # 读取JSON文件

    key = f"topic:{topic}"          # 构建查询键topic:<topic>，检查该键是否存在于JSON数据中。
    if key not in data_dict:
        print(f"ERROR: Tactile data for topic '{topic}' not found in {json_path}")
        return [], []     # 若不存在，打印错误信息并返回空列表。

    # 提取原始数据
    sub = data_dict[key]     
    times_str = sub["times"]    # 获取时间信息
    values_raw = sub["values"]  # 获取原始值

    """时间戳解析与数据对齐 """
    for (t_str, v) in zip(times_str, values_raw):   # 遍历时间和数值的配对；zip是按“按位置配对” 特性；将两个列表按 “相同索引位置” 进行配对
        try:
            dt = iso_or_epoch_to_datetime(t_str)    # 调用 iso_or_epoch_to_datetime 将时间字符串 t_str 转换为 datetime 对象。 
            times.append(dt)                        # 支持格式：ISO 8601（如 "2023-01-01T00:00:00Z"）或Unix时间戳（如 1640995200）。
            values.append(v)
        except Exception as ex:
            print(f"Failed to parse local_time {t_str}: {ex}")
            continue

    # Sort by ascending datetime
    combined = sorted(zip(times, values), key=lambda x: x[0])   # 将时间和数值合并为元组列表 combined。按时间升序排序（key=lambda x: x[0]）。
    if not combined:                                            # 若 combined 为空，返回空列表。
        print(f"No data found in {json_path} for {topic}.")
        return [], []
    times_sorted, values_sorted = zip(*combined)                # 解压与返回：使用 zip(*combined) 解压排序后的时间和数值。
    return list(times_sorted), list(values_sorted)              # 转换为列表返回（times_sorted, values_sorted）。

""" 
    参数：
        tactile_times：已排序的触觉数据时间戳列表（datetime 对象，必须升序）。
        target_time：目标时间点（datetime 对象）。
    返回：
        返回 int 类型的索引（满足条件的最近时间点索引），或 None（未找到有效索引）。
"""
def find_latest_index(tactile_times, target_time):
    i = bisect_right(tactile_times, target_time)    # 在有序列表 tactile_times 中查找 target_time 的插入位置 i，使得插入后列表仍有序。
    idx = i - 1                                     # idx = i - 1 表示 tactile_times 中 最后一个小于等于 target_time 的时间点索引。
    if idx < 0:                                     # 当 target_time 比 tactile_times 中所有时间都早时，i = 0，idx = -1。
        return None
    if target_time - tactile_times[idx] > timedelta(seconds=5): # 检查找到的时间点 tactile_times[idx] 是否与 target_time 相差超过5秒。
        return None
    return idx

def process_cut_video(
    cut_video_path,     # 剪切后的视频路径 
    base_timestamp,     # 基准时间戳（datetime对象）
    base_frame_index,   # 基准帧号（整数）
    cut_start_index,    # 剪切起始帧号（整数）
    fps,                # 视频帧率
    left_times,         # 左侧触觉数据的时间戳
    left_values,        # 左侧触觉数据的数值列表
    right_times,        # 右侧触觉数据的时间戳
    right_values,       # 右侧触觉数据的数值列表
    npy_save_path,      # 触觉数据保存路径（.npy文件）
    plot_video=False,   # 是否生成叠加视频
    plot_images=False   # 是否保存叠加帧图像 
):
    """
    Reads the *cut* video. For each frame i, compute abs_time and find nearest tactile frames.
    Save .npy with shape (num_frames, 12, 64). If either plot_video=True or plot_images=True,
    we create an overlay of camera + tactile, and optionally:
      - if plot_video=True, write that overlay to an MP4
      - if plot_images=True, save each overlay frame to frames/*.png
    When plot_video=True, also save a separate video with only the tactile frames,
    with a small gap between left and right.

    输入：剪切视频
    时间计算：对每一帧i，计算其绝对时间abs_time（基于 base_timestamp 和帧号偏移）。
    数据对齐：为每帧找到左右触觉数据中时间最近的匹配点（使用 find_latest_index）。
    数据保存：将触觉数据保存为 .npy 文件，形状为 (num_frames, 12, 64)。
        这里（12，64）是左右触觉数据的拼接，如果我使用我的话，我就要做一个（12，32）的。
    一旦，当 plot_video 或 plot_images 为 True 时，会生成视频帧与触觉数据的叠加图像。
    叠加内容：
        上方：视频帧（缩放至与触觉图同宽度）。
        下方：左右触觉数据的伪彩色图（COLORMAP_VIRIDIS），中间用 gap_size 像素的空白隔开。
    输出形式：
        视频文件（plot_video=True）：生成一个MP4文件，包含所有叠加帧。（当plot_video = True，也会输出一个纯触觉帧，左右触觉图像之间保留一个gap_size的间隔）
        单帧图像（plot_images=True）：将每帧叠加图保存为 frames/ 目录下的PNG文件。
    """
    cap = cv2.VideoCapture(cut_video_path)
    if not cap.isOpened():      # 打开视频文件
        print(f"Cannot open cut video {cut_video_path}")
        return 0

    # DEBUG PRINT: check how many frames the cut video has (per cv2)
    # cut_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"DEBUG: The newly cut video '{cut_video_path}' reports {cut_total_frames} frames via cv2.")

    all_tactile = []    # 初始化存储触觉数据的列表
    frame_count = 0     # 帧计数器

    video_writer = None     # 视频写入器
    tactile_writer = None   # 触觉写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # 指定视频编码格式为'mp4v' ，是一种常见的视频编码格式（MPEG-4 编码），用于生成 .mp4 格式的视频文件。

    # 可视化相关初始化
    do_overlay = plot_video or plot_images  # 判断是否要生成叠加图像
    frames_dir = None                       # 初始化帧文件夹
    if plot_images:
        frames_dir = os.path.join(os.path.dirname(npy_save_path), "frames") # 设定帧保存路径
        os.makedirs(frames_dir, exist_ok=True)

    # Define gap size in pixels between left and right tactile visuals
    gap_size = 20

    while True:
        ret, frame = cap.read()     # cap.read()每次都只读取一帧。其中ret是一个布尔值，表示是否成功读取到帧。
        if not ret:                 # frame 是读取到的那一帧图像（numpy 数组格式，包含像素数据）
            break
        frame_count += 1            # 记录帧数字+1

        absolute_time = base_timestamp + timedelta(     # 计算当前帧的绝对时间
            seconds=((cut_start_index + frame_count) - base_frame_index) / fps
        )

        idx_left = find_latest_index(left_times, absolute_time) # 找到左侧触觉数据最近的时间点
        if idx_left is None:    # 如果找不到，就立刻释放资源
            print(f"No tactile match (left) at frame {frame_count} => {absolute_time}. Aborting video.")
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if tactile_writer is not None:
                tactile_writer.release()
            return 0

        idx_right = find_latest_index(right_times, absolute_time) # 找到右侧触觉数据最近的时间点
        if idx_right is None:    # 如果找不到，就立刻释放资源
            print(f"No tactile match (right) at frame {frame_count} => {absolute_time}. Aborting video.")
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if tactile_writer is not None:
                tactile_writer.release()
            return 0

        # 如果说我直接不改，行不行，可能我给的是12*16，但是他能不能给我自动补齐到（12，32）中，当然补齐合不合适，最好还是用（12，16）
        arr_left = np.array(left_values[idx_left], dtype=np.float32).reshape((12, 32))
        arr_right = np.array(right_values[idx_right], dtype=np.float32).reshape((12, 32))
        combined = np.hstack((arr_left, arr_right))  # shape (12,64)
        all_tactile.append(combined)

        if do_overlay:  # 确定要生成叠加图像
            # 将触觉数据归一化到[0, 255]
            left_vis_u8 = (arr_left * 255).astype(np.uint8)
            right_vis_u8 = (arr_right * 255).astype(np.uint8)
            left_color = cv2.applyColorMap(left_vis_u8, cv2.COLORMAP_VIRIDIS)
            right_color = cv2.applyColorMap(right_vis_u8, cv2.COLORMAP_VIRIDIS)

            scale_factor = 10   # 放大倍数
            left_color_big = cv2.resize(    # 将图像放大10倍
                left_color,
                (left_color.shape[1]*scale_factor, left_color.shape[0]*scale_factor),
                interpolation=cv2.INTER_NEAREST
            )
            right_color_big = cv2.resize(    # 将图像放大10倍
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

            final_img = np.vstack((cam_resized, tactile_vis))   # 垂直叠加视频帧和触觉图

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

    # 保存触觉数据
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
    # 截断相机轨迹文件
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
    vid,                        # 输入视频路径（通常为 raw_video.mp4 ）
    final_output_dir,           # 输出目录路径 
    left_times, left_values,    # 左侧触觉数据时间戳和数值列表 
    right_times, right_values,  # 右侧触觉数据时间戳和数值列表
    black_threshold,            # 黑帧检测的灰度阈值 
    cut_delay,                  # 剪切延迟时间（秒）
    qr_latency,                 # QR码时间戳补偿值（秒）
    plot_video,                 # 是否生成叠加视频
    plot_images=False           # 是否保存叠加图像 
):
    parent_dir = os.path.dirname(vid)       # 获取父目录路径
    original_raw_path = os.path.join(parent_dir, "original_raw_video.mp4")  # 构建目标文件路径；os.path.join() 将父目录路径与目标文件名 original_raw_video.mp4 拼接，生成完整的目标文件路径。
    if os.path.exists(original_raw_path):   # 检查目标文件是否已存在
        vid = original_raw_path
        print("Found original_raw_video.mp4, skipping rename.")
    else:                                   # 如果目标文件不存在，则执行重命名
        try:
            os.rename(vid, original_raw_path)
            vid = original_raw_path
            print("Renamed raw_video.mp4 => original_raw_video.mp4")
        except Exception as e:
            print(f"Cannot rename {vid} => {original_raw_path}, error: {e}")
            return False
    
    """获取视频帧率和总帧数，并立即释放资源"""
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"Cannot open {vid}, skipping.")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {vid} => FPS={fps:.3f}, total frames={total_frames}")

    """解析视频中的OR码（时间戳和帧号）和黑帧（帧号）若没有OR码或黑帧，终止处理"""
    base_ts, base_idx, black_idx = scan_qr_and_black(vid, qr_latency, black_threshold)
    if base_ts is None or base_idx is None:
        print("No valid QR code found => skipping.")
        return False
    if black_idx is None:
        print("No black frame found => skipping.")
        return False

    # 计算剪切位置，但并不剪切
    cut_time = (black_idx / fps) + cut_delay
    cut_start_index = int(round(cut_time * fps))
    print(f"Cut time = {cut_time:.2f}s => frame idx ~{cut_start_index}")

    # 输出视频路径
    out_cut_mp4 = os.path.join(final_output_dir, "raw_video.mp4")
    success = cut_video_ffmpeg(vid, out_cut_mp4, cut_time)  # 视频剪切
    if not success:
        print("Cut failed => skipping.")                    # 剪切失败
        return False

    # debug_cap = cv2.VideoCapture(out_cut_mp4)
    # debug_cut_frames = int(debug_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # debug_cap.release()
    # print(f"DEBUG: Just cut '{out_cut_mp4}', it has {debug_cut_frames} frames per cv2.")

    out_npy = os.path.join(final_output_dir, "tactile.npy")
    final_count = process_cut_video(        # 对齐剪切后视频的每一帧与触觉数据，生成 tactile.npy （形状 (N,12,64)）。可选生成叠加视频（plot_video）或单帧图像（plot_images）。
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

    # IMU数据截断（可选）
    """
        主要功能是处理 IMU（惯性测量单元）数据的 JSON 文件，根据之前计算的裁剪时间（cut_time）
        对 IMU 数据进行修剪和时间校准，确保其与裁剪后的视频数据在时间上保持同步。
    """
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
            video_list.append(item)     # video_list 包含所有待处理的MP4文件绝对路径。
        else:
            print(f"WARNING: {item} is not a file or directory, skipping.")

    if not video_list:                  # 如果video list为空，输出无法找到mp4文件，之后退出。
        print("No .mp4 files found.")
        sys.exit(1)

    left_times, left_values = load_tactile_data(args.bag, args.tactile_topic_left)      # 从JSON文件中提取指定主题的时间戳和数值列表。
    right_times, right_values = load_tactile_data(args.bag, args.tactile_topic_right)   # 从JSON文件中提取指定主题的时间戳和数值列表。
    if (not left_times) and (not right_times):                                          # 若左右触觉数据均为空，直接退出程序。
        print("No tactile messages for either topic => exit.")
        sys.exit(1)

    success_count = 0       # 成功计数器
    fail_count = 0          # 失败计数器

    with ProcessPoolExecutor(max_workers=2) as executor:        # 使用 ProcessPoolExecutor 创建2个 worker 进程（max_workers=2）。
        futures = []
        for vid in video_list:
            futures.append(
                executor.submit(        # 通过 executor.submit 提交异步任务，返回 Future 对象。
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
    print(f"Successful videos: {success_count}, Failed videos: {fail_count}")   # 输出处理完成的总结信息，包括成功和失败的数量。

if __name__ == "__main__":
    main()
