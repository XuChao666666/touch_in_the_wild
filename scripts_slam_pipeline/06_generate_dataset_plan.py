#!/usr/bin/env python3

# 06_generate_dataset_plan.py
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import pickle
import numpy as np
import json
import math
import collections
import scipy.ndimage as sn
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.cv_util import (
    get_gripper_width
)
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator, 
    get_interp1d,
    PoseInterpolator
)

# %%
def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i+1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type

def pose_interp_from_df(df, start_timestamp=0.0, tx_base_slam=None):    
    timestamp_sec = df['timestamp'].to_numpy() + start_timestamp
    cam_pos = df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()
    tx_slam_cam = cam_pose
    tx_base_cam = tx_slam_cam
    if tx_base_slam is not None:
        tx_base_cam = tx_base_slam @ tx_slam_cam
    pose_interp = PoseInterpolator(
        t=timestamp_sec, x=mat_to_pose(tx_base_cam))
    return pose_interp

def get_x_projection(tx_tag_this, tx_tag_other):
    t_this_other = tx_tag_other[:,:3,3] - tx_tag_this[:,:3,3]
    v_this_forward = tx_tag_this[:,:3,2]
    v_up = np.array([0.,0.,1.])
    v_this_right = np.cross(v_this_forward, v_up)
    proj_other_right = np.sum(v_this_right * t_this_other, axis=-1)
    return proj_other_right

"""
    函数功能：将SLAM系统的输出gripper的宽度线性映射到xArm的控制范围，同时包含了边界限制，确保输入值在有效范围内；
    输入：SLAM输出的宽度；SLAM最大宽度和最小宽度；
    输出：映射到xArm下的结果；
    个人猜测：SLAM下的宽度应该是米为单位；xArm下的应该是cm为单位；
    所以这里映射一下，单纯从数值来讲，差距并不大；
"""
def slam_to_xarm(width, SLAM_MAX_WIDTH, SLAM_MIN_WIDTH):
    """
    Linearly map a raw (SLAM) gripper width in [0.081, 0.168]
    to xArm control range [-0.01, 0.842].
    Clamps width to [SLAM_GRIPPER_MIN, SLAM_GRIPPER_MAX].
    """
    XARM_GRIPPER_MIN = -0.01    # XARM夹爪的宽度最小值
    XARM_GRIPPER_MAX = 0.800    # XARM夹爪的宽度最大值

    w = max(SLAM_MIN_WIDTH, min(SLAM_MAX_WIDTH, width)) # 确保width在SLAM_MIN 和 SLAM_MAX中
    # Linear interpolation formula
    return ((w - SLAM_MIN_WIDTH) / (SLAM_MAX_WIDTH - SLAM_MIN_WIDTH)) \
           * (XARM_GRIPPER_MAX - XARM_GRIPPER_MIN) + XARM_GRIPPER_MIN
"""
    与UMI的差别：
        1、输入参数tcp_offset：这里的值是0.171，UMI是0.205，我应该也是要用0.205
        2、多出一个slam_to_xarm函数，这里需要和UMI做一个详细比对；
        3、文件注释头：#!/usr/bin/env python3  很多文件都有这个头，作者这里是强制使用系统的python解释器，但是我这里应该会使用虚拟环境的python解释器，所以这里必须要修改的。
        4、cam_to_center_height不同，UMI中该值为0.086；这里是0.082；还是要做一些调整的；
        5、这里从original_raw_video.mp4中读取，而UMI中读取的文件是raw_video.mp4 
        6、touch中继承之前的tcp_offset=0.171；UMI中继承之前的0.205
        7、这里gripper_vid_idxs = [] 和 pose_interps = []；UMI中是gripper_vid_idxs = list() 和 pose_interps = list()
        8、夹爪宽度计算方法：这里使用slam_to_xarm();UMI中使用校准插值器gripper_cal_interp(width)；这里要做一下区别；
        9、初始化列表：touch中初始化用的[],UMI中用的是list（）；（区别：[]只可以定义列表；list（）可以将其他任何形式转换为列表，例如元组）
        10、
"""
# %%
@click.command()
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-o', '--output', default=None)
@click.option('-to', '--tcp_offset', type=float, default=0.171, help="Distance from gripper tip to mounting screw") #0.205 - 0.043
@click.option('-ts', '--tx_slam_tag', default=None, help="tx_slam_tag.json")
@click.option('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag") #0.072 - 0.043 + 0.015
@click.option('-ml', '--min_episode_length', type=int, default=24)
@click.option('--ignore_cameras', type=str, default=None, help="comma separated string of camera serials to ignore")
def main(input, output, tcp_offset, tx_slam_tag,
         nominal_z, min_episode_length, ignore_cameras):
    # %% stage 0
    # gather inputs
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('dataset_plan.pkl')

    # tcp to camera transform
    # all unit in meters
    # y axis in camera frame
    cam_to_center_height = 0.082 # constant for UMI 
    # optical center to mounting screw, positive is when optical center is in front of the mount
    cam_to_mount_offset = 0.01465 # constant for GoPro Hero 9,10,11
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset

    pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0])
    tx_cam_tcp = pose_to_mat(pose_cam_tcp)
        
    # SLAM map origin to table tag transform
    if tx_slam_tag is None:
        path = demos_dir.joinpath('mapping', 'tx_slam_tag.json')
        assert path.is_file()
        tx_slam_tag = str(path)
    tx_slam_tag = np.array(json.load(
        open(os.path.expanduser(tx_slam_tag), 'r')
        )['tx_slam_tag']
    )
    tx_tag_slam = np.linalg.inv(tx_slam_tag)

    # load gripper calibration
    gripper_id_gripper_cal_map = dict()
    cam_serial_gripper_cal_map = dict()

    with ExifToolHelper() as et:
        for gripper_cal_path in demos_dir.glob("gripper*/gripper_range.json"):
            mp4_path = gripper_cal_path.parent.joinpath('raw_video.mp4')
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta['QuickTime:CameraSerialNumber']

            gripper_range_data = json.load(gripper_cal_path.open('r'))
            gripper_id = gripper_range_data['gripper_id']
            max_width = gripper_range_data['max_width']
            min_width = gripper_range_data['min_width']
            gripper_cal_data = {
                'aruco_measured_width': [min_width, max_width],
                'aruco_actual_width': [min_width, max_width]
            }
            gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)
            gripper_id_gripper_cal_map[gripper_id] = gripper_cal_interp
            cam_serial_gripper_cal_map[cam_serial] = gripper_cal_interp

    
    # %% stage 1
    # loop over all demo directory to extract video metadata
    # output: video_meta_df
    
    # find videos
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])

    # ignore camera
    ignore_cam_serials = set()
    if ignore_cameras is not None:
        serials = ignore_cameras.split(',')
        ignore_cam_serials = set(serials)
    
    fps = None
    rows = list()
    with ExifToolHelper() as et:
        for video_dir in video_dirs: 
            mp4_path = video_dir.joinpath('raw_video.mp4')
            original_mp4_path = video_dir.joinpath('original_raw_video.mp4')
            meta = list(et.get_metadata(str(original_mp4_path)))[0]
            cam_serial = meta['QuickTime:CameraSerialNumber']
            start_date = mp4_get_start_datetime(str(original_mp4_path))
            start_timestamp = start_date.timestamp()

            if cam_serial in ignore_cam_serials:
                print(f"Ignored {video_dir.name}")
                continue
            
            csv_path = video_dir.joinpath('camera_trajectory.csv')
            if not csv_path.is_file():
                print(f"Ignored {video_dir.name}, no camera_trajectory.csv")
                continue
            
            pkl_path = video_dir.joinpath('tag_detection.pkl')
            if not pkl_path.is_file():
                print(f"Ignored {video_dir.name}, no tag_detection.pkl")
                continue
            
            with av.open(str(mp4_path), 'r') as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                if fps is None:
                    fps = stream.average_rate
                else:
                    if fps != stream.average_rate:
                        print(f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}")
                        exit(1)
            duration_sec = float(n_frames / fps)
            end_timestamp = start_timestamp + duration_sec
            
            rows.append({
                'video_dir': video_dir,
                'camera_serial': cam_serial,
                'start_date': start_date,
                'n_frames': n_frames,
                'fps': fps,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp
            })
    if len(rows) == 0:
        print("No valid videos found!")
        exit(1)
            
    video_meta_df = pd.DataFrame(data=rows)

    # %% stage 2
    # match videos into demos
    serial_count = video_meta_df['camera_serial'].value_counts()
    print("Found following cameras:")
    print(serial_count)
    n_cameras = len(serial_count)
    
    events = list()
    for vid_idx, row in video_meta_df.iterrows():
        events.append({
            'vid_idx': vid_idx,
            'camera_serial': row['camera_serial'],
            't': row['start_timestamp'],
            'is_start': True
        })
        events.append({
            'vid_idx': vid_idx,
            'camera_serial': row['camera_serial'],
            't': row['end_timestamp'],
            'is_start': False
        })
    events = sorted(events, key=lambda x: x['t'])
    
    demo_data_list = list()
    on_videos = set()
    on_cameras = set()
    used_videos = set()
    t_demo_start = None
    for i, event in enumerate(events):
        if event['is_start']:
            on_videos.add(event['vid_idx'])
            on_cameras.add(event['camera_serial'])
        else:
            on_videos.remove(event['vid_idx'])
            on_cameras.remove(event['camera_serial'])
        assert len(on_videos) == len(on_cameras)
        
        if len(on_cameras) == n_cameras:
            # start demo episode
            t_demo_start = event['t']
        elif t_demo_start is not None:
            # stop the episode
            assert not event['is_start']
            t_start = t_demo_start
            t_end = event['t']
            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(event['vid_idx'])
            used_videos.update(demo_vid_idxs)
            demo_data_list.append({
                "video_idxs": sorted(demo_vid_idxs),
                "start_timestamp": t_start,
                "end_timestamp": t_end
            })
            t_demo_start = None
    unused_videos = set(video_meta_df.index) - used_videos
    for vid_idx in unused_videos:
        print(f"Warning: video {video_meta_df.loc[vid_idx]['video_dir'].name} unused in any demo")

    # %% stage 3
    # identify gripper id (hardware) using aruco
    finger_tag_det_th = 0.8
    vid_idx_gripper_hardware_id_map = dict()
    cam_serial_gripper_ids_map = collections.defaultdict(list)
    for vid_idx, row in video_meta_df.iterrows():
        video_dir = row['video_dir']
        pkl_path = video_dir.joinpath('tag_detection.pkl')
        if not pkl_path.is_file():
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            continue
        tag_data = pickle.load(pkl_path.open('rb'))
        n_frames = len(tag_data)
        tag_counts = collections.defaultdict(lambda: 0)
        for frame in tag_data:
            for key in frame['tag_dict'].keys():
                tag_counts[key] += 1
        tag_stats = collections.defaultdict(lambda: 0.0)
        for k, v in tag_counts.items():
            tag_stats[k] = v / n_frames
            
        max_tag_id = np.max(list(tag_stats.keys()))
        tag_per_gripper = 6
        max_gripper_id = max_tag_id // tag_per_gripper
        
        gripper_prob_map = dict()
        for gripper_id in range(max_gripper_id+1):
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1
            left_prob = tag_stats[left_id]
            right_prob = tag_stats[right_id]
            gripper_prob = min(left_prob, right_prob)
            if gripper_prob <= 0:
                continue
            gripper_prob_map[gripper_id] = gripper_prob
        
        gripper_id_by_tag = -1
        if len(gripper_prob_map) > 0:
            gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[-1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]
            if gripper_prob >= finger_tag_det_th:
                gripper_id_by_tag = gripper_id

        cam_serial_gripper_ids_map[row['camera_serial']].append(gripper_id_by_tag)
        vid_idx_gripper_hardware_id_map[vid_idx] = gripper_id_by_tag
    
    series = pd.Series(
        data=list(vid_idx_gripper_hardware_id_map.values()), 
        index=list(vid_idx_gripper_hardware_id_map.keys()))
    video_meta_df['gripper_hardware_id'] = series
    
    cam_serial_gripper_hardware_id_map = dict()
    for cam_serial, gripper_ids in cam_serial_gripper_ids_map.items():
        counter = collections.Counter(gripper_ids)
        if len(counter) != 1:
            print(f"warning: multiple gripper ids {counter} detected for camera serial {cam_serial}")
        gripper_id = counter.most_common()[0][0]
        cam_serial_gripper_hardware_id_map[cam_serial] = gripper_id 

    # %% stage 4
    # disambiguiate gripper left/right
    n_gripper_cams = (np.array(list(
        cam_serial_gripper_hardware_id_map.values())
        ) >= 0).sum()

    if n_gripper_cams <= 0:
        # no gripper camera
        raise RuntimeError("No gripper camera detected!")

    grip_cam_serials = []
    other_cam_serials = []
    for cs, gi in cam_serial_gripper_hardware_id_map.items():
        if gi >= 0:
            grip_cam_serials.append(cs)
        else:
            other_cam_serials.append(cs)
    
    cam_serial_cam_idx_map = {}
    for i, cs in enumerate(sorted(other_cam_serials)):
        cam_serial_cam_idx_map[cs] = len(grip_cam_serials) + i

    cam_serial_right_to_left_idx_map = collections.defaultdict(list)
    vid_idx_cam_idx_map = np.full(len(video_meta_df), fill_value=-1, dtype=np.int32)
    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data['video_idxs']
        start_timestamp = demo_data['start_timestamp']
        end_timestamp = demo_data['end_timestamp']

        cam_serials = []
        gripper_vid_idxs = []
        pose_interps = []

        for vid_idx in video_idxs:
            row = video_meta_df.loc[vid_idx]
            if row.gripper_hardware_id < 0:
                cam_serial = row['camera_serial']
                if cam_serial in cam_serial_cam_idx_map:
                    vid_idx_cam_idx_map[vid_idx] = cam_serial_cam_idx_map[cam_serial]
                continue
            cam_serials.append(row['camera_serial'])
            gripper_vid_idxs.append(vid_idx)
            vid_dir = row['video_dir']
                        
            csv_path = vid_dir.joinpath('camera_trajectory.csv')
            if not csv_path.is_file():
                break

            csv_df = pd.read_csv(csv_path)
            if csv_df['is_lost'].sum() > 10:
                break
            if (~csv_df['is_lost']).sum() < 60:
                break

            df = csv_df.loc[~csv_df['is_lost']]
            pose_interp = pose_interp_from_df(df, 
                start_timestamp=row['start_timestamp'], 
                tx_base_slam=tx_tag_slam)
            pose_interps.append(pose_interp)
        
        if len(pose_interps) != n_gripper_cams:
            print(f"Excluded demo {demo_idx} from left/right disambiguation.")
            continue
        
        n_samples = 100
        t_samples = np.linspace(start_timestamp, end_timestamp, n_samples)
        pose_samples = [pose_to_mat(interp(t_samples)) for interp in pose_interps]
        
        x_proj_avg = []
        for i in range(len(pose_samples)):
            this_proj_avg = []
            for j in range(len(pose_samples)):
                this_proj_avg.append(np.mean(get_x_projection(
                    tx_tag_this=pose_samples[i], 
                    tx_tag_other=pose_samples[j])))
            this_proj_avg = np.mean(this_proj_avg)
            x_proj_avg.append(this_proj_avg)
        
        camera_right_to_left_idxs = np.argsort(x_proj_avg)
        
        for vid_idx, cam_serial, cam_right_idx in zip(
            gripper_vid_idxs, cam_serials, camera_right_to_left_idxs):
            cam_serial_right_to_left_idx_map[cam_serial].append(cam_right_idx)
            vid_idx_cam_idx_map[vid_idx] = cam_right_idx

    for cs, cis in cam_serial_right_to_left_idx_map.items():
        count = collections.Counter(cis)
        this_cam_idx = count.most_common(1)[0][0]
        cam_serial_cam_idx_map[cs] = this_cam_idx

    camera_idx_series = video_meta_df['camera_serial'].map(cam_serial_cam_idx_map)
    camera_idx_from_episode_series = pd.Series(
        data=vid_idx_cam_idx_map, 
        index=video_meta_df.index)
    
    video_meta_df['camera_idx'] = camera_idx_series
    video_meta_df['camera_idx_from_episode'] = camera_idx_from_episode_series
    
    rows = []
    for cs, ci in cam_serial_cam_idx_map.items():
        rows.append({
            'camera_idx': ci,
            'camera_serial': cs,
            'gripper_hw_idx': cam_serial_gripper_hardware_id_map[cs],
            'example_vid': video_meta_df.loc[video_meta_df['camera_serial'] == cs].iloc[0]['video_dir'].name
        })
    camera_serial_df = pd.DataFrame(data=rows)
    camera_serial_df.set_index('camera_idx', inplace=True)
    camera_serial_df.sort_index(inplace=True)
    print("Assigned camera_idx: right=0; left=1; non_gripper=2,3...")
    print(camera_serial_df)
    
    # %% stage 6
    total_avaliable_time = 0.0
    total_used_time = 0.0
    dropped_camera_count = collections.defaultdict(lambda: 0)
    n_dropped_demos = 0
    all_plans = []
    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data['video_idxs']
        start_timestamp = demo_data['start_timestamp']
        end_timestamp = demo_data['end_timestamp']
        total_avaliable_time += (end_timestamp - start_timestamp)

        demo_video_meta_df = video_meta_df.loc[video_idxs].copy()
        demo_video_meta_df.set_index('camera_idx', inplace=True)
        demo_video_meta_df.sort_index(inplace=True)
        
        dt = None
        alignment_costs = []
        for cam_idx, row in demo_video_meta_df.iterrows():
            dt = 1 / row['fps']
            this_alignment_cost = []
            for other_cam_idx, other_row in demo_video_meta_df.iterrows():
                diff = other_row['start_timestamp'] - row['start_timestamp']
                remainder = diff % dt
                this_alignment_cost.append(remainder)
            alignment_costs.append(this_alignment_cost)

        align_cam_idx = np.argmin([sum(x) for x in alignment_costs])
        align_video_start = demo_video_meta_df.loc[align_cam_idx]['start_timestamp']
        start_timestamp += dt - ((start_timestamp - align_video_start) % dt)
        cam_start_frame_idxs = []
        n_frames = int((end_timestamp - start_timestamp) / dt)
        for cam_idx, row in demo_video_meta_df.iterrows():
            video_start_frame = math.ceil((start_timestamp - row['start_timestamp']) / dt)
            video_n_frames = math.floor((row['end_timestamp'] - start_timestamp) / dt) - 1
            if video_start_frame < 0:
                video_n_frames += video_start_frame
                video_start_frame = 0
            cam_start_frame_idxs.append(video_start_frame)
            n_frames = min(n_frames, video_n_frames)
        demo_timestamps = np.arange(n_frames) * float(dt) + start_timestamp

        all_cam_poses = []
        all_gripper_widths = []
        all_is_valid = []
        
        for cam_idx, row in demo_video_meta_df.iterrows():
            if cam_idx >= n_gripper_cams:
                continue  # not a gripper camera

            start_frame_idx = cam_start_frame_idxs[cam_idx]
            video_dir = row['video_dir']
            
            check_path = video_dir.joinpath('check_result.txt')
            if check_path.is_file():
                if not check_path.open('r').read().startswith('true'):
                    print(f"Skipping {video_dir.name}, manually filtered with check_result.txt!=true")
                    continue

            csv_path = video_dir.joinpath('camera_trajectory.csv')
            if not csv_path.is_file():
                print(f"Skipping {video_dir.name}, no camera_trajectory.csv.")
                dropped_camera_count[row['camera_serial']] += 1
                continue            
            
            csv_df = pd.read_csv(csv_path)
            df = csv_df.iloc[start_frame_idx: start_frame_idx+n_frames]
            is_tracked = (~df['is_lost']).to_numpy()
            n_frames_lost = (~is_tracked).sum()
            if n_frames_lost > 10:
                print(f"Skipping {video_dir.name}, {n_frames_lost} frames are lost.")
                dropped_camera_count[row['camera_serial']] += 1
                continue

            n_frames_valid = is_tracked.sum()
            if n_frames_valid < 60:
                print(f"Skipping {video_dir.name}, only {n_frames_valid} frames are valid.")
                dropped_camera_count[row['camera_serial']] += 1
                continue
            
            df.loc[df['is_lost'], 'q_w'] = 1
            cam_pos = df[['x', 'y', 'z']].to_numpy()
            cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
            cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
            cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
            cam_pose[:,3,3] = 1
            cam_pose[:,:3,3] = cam_pos
            cam_pose[:,:3,:3] = cam_rot.as_matrix()
            tx_slam_cam = cam_pose
            tx_tag_cam = tx_tag_slam @ tx_slam_cam

            is_step_valid = is_tracked.copy()

            pkl_path = video_dir.joinpath('tag_detection.pkl')
            if not pkl_path.is_file():
                print(f"Skipping {video_dir.name}, no tag_detection.pkl.")
                dropped_camera_count[row['camera_serial']] += 1
                continue
                        
            tag_detection_results = pickle.load(open(pkl_path, 'rb'))
            tag_detection_results = tag_detection_results[start_frame_idx: start_frame_idx+n_frames]
            video_timestamps = np.array([x['time'] for x in tag_detection_results])
            if len(df) != len(video_timestamps):
                print(f"Skipping {video_dir.name}, video csv length mismatch.")
                continue

            ghi = row['gripper_hardware_id']
            if ghi < 0:
                print(f"Skipping {video_dir.name}, invalid gripper hardware id {ghi}")
                dropped_camera_count[row['camera_serial']] += 1
                continue
            
            left_id = 6 * ghi
            right_id = left_id + 1

            gripper_cal_interp = None
            if ghi in gripper_id_gripper_cal_map:
                gripper_cal_interp = gripper_id_gripper_cal_map[ghi]
            elif row['camera_serial'] in cam_serial_gripper_cal_map:
                gripper_cal_interp = cam_serial_gripper_cal_map[row['camera_serial']]
                print(f"Gripper id {ghi} not found in calibrations. Falling back to camera serial map.")
            else:
                raise RuntimeError("Gripper calibration not found.")

            gripper_timestamps = []
            gripper_widths = []
            for td in tag_detection_results:
                width = get_gripper_width(td['tag_dict'], 
                                          left_id=left_id, 
                                          right_id=right_id, 
                                          nominal_z=nominal_z)
                
                if width is not None:
                    # Instead of "gripper_cal_interp(width)", we do:
                    w_xarm = slam_to_xarm(width, max_width, min_width)
                    #print("after processed",w_xarm)
                    gripper_timestamps.append(td['time'])
                    gripper_widths.append(w_xarm)

                '''
                if width is not None:
                    gripper_timestamps.append(td['time'])
                    gripper_widths.append(gripper_cal_interp(width))
                '''
            gripper_interp = get_interp1d(gripper_timestamps, gripper_widths)
            gripper_det_ratio = (len(gripper_widths) / len(tag_detection_results))
            if gripper_det_ratio < 0.9:
                print(f"Warining: {video_dir.name} only {gripper_det_ratio} of gripper tags detected.")
            
            this_gripper_widths = gripper_interp(video_timestamps)

            # -------------------------------
            # Debug print: Print gripper width for each frame
            #print(f"\nGripper widths in {video_dir.name}:")
            #for i, w in enumerate(this_gripper_widths):
            #    print(f"  Frame {start_frame_idx + i}: gripper_width = {w}")
            # -------------------------------

            tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
            pose_tag_tcp = mat_to_pose(tx_tag_tcp)
            
            assert len(pose_tag_tcp) == n_frames
            assert len(this_gripper_widths) == n_frames
            assert len(is_step_valid) == n_frames
            all_cam_poses.append(pose_tag_tcp)
            all_gripper_widths.append(this_gripper_widths)
            all_is_valid.append(is_step_valid)

        if len(all_cam_poses) != n_gripper_cams:
            print(f"Skipped demo {demo_idx}.")
            n_dropped_demos += 1
            continue

        all_is_valid = np.array(all_is_valid)
        is_step_valid = np.all(all_is_valid, axis=0)
        
        first_valid_step = np.nonzero(is_step_valid)[0][0]
        last_valid_step = np.nonzero(is_step_valid)[0][-1]
        demo_start_poses = []
        demo_end_poses = []
        for cam_idx in range(len(all_cam_poses)):
            cam_poses = all_cam_poses[cam_idx]
            demo_start_poses.append(cam_poses[first_valid_step])
            demo_end_poses.append(cam_poses[last_valid_step])

        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid_segment in zip(segment_slices, segment_type):
            start = s.start
            end = s.stop
            if not is_valid_segment:
                continue
            if (end - start) < min_episode_length:
                is_step_valid[start:end] = False
        
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid in zip(segment_slices, segment_type):
            if not is_valid:
                continue
            start = s.start
            end = s.stop
            total_used_time += float((end - start) * dt)
            
            grippers = []
            cameras = []
            for cam_idx, row in demo_video_meta_df.iterrows():
                if cam_idx < n_gripper_cams:
                    pose_tag_tcp = all_cam_poses[cam_idx][start:end]
                    grippers.append({
                        "tcp_pose": pose_tag_tcp,
                        "gripper_width": all_gripper_widths[cam_idx][start:end],
                        "demo_start_pose": demo_start_poses[cam_idx],
                        "demo_end_pose": demo_end_poses[cam_idx]
                    })
                video_dir = row['video_dir']
                vid_start_frame = cam_start_frame_idxs[cam_idx]
                cameras.append({
                    "video_path": str(video_dir.joinpath('raw_video.mp4').relative_to(video_dir.parent)),
                    "video_start_end": (start+vid_start_frame, end+vid_start_frame)
                })
            
            all_plans.append({
                "episode_timestamps": demo_timestamps[start:end],
                "grippers": grippers,
                "cameras": cameras
            })

    used_ratio = total_used_time / total_avaliable_time
    print(f"{int(used_ratio*100)}% of raw data are used.")

    print(dropped_camera_count)
    print("n_dropped_demos", n_dropped_demos)

    # %%
    pickle.dump(all_plans, output.open('wb'))
    

def test():
    pass

if __name__ == "__main__":
    main()
