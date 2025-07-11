#!/usr/bin/env python3
# 06_generate_dataset_plan.py
# %%
import sys
import os
import pathlib
import click
import pickle
import numpy as np
import json
import math
import collections
import scipy.ndimage as sn
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# -------------------------------------------------------------------------- #
# ExifTool is optional; if it's missing we produce PLACEHOLDER values instead
# -------------------------------------------------------------------------- #
try:
    from exiftool import ExifToolHelper
except ImportError:
    ExifToolHelper = None

from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.cv_util import get_gripper_width
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator,
    get_interp1d,
    PoseInterpolator
)

# -------------------------------------------------------------------------- #
#                               UTILITIES                                    #
# -------------------------------------------------------------------------- #
def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = []
    segment_type = []
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i + 1]
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
    cam_pose[:, 3, 3] = 1
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()
    tx_slam_cam = cam_pose
    tx_base_cam = tx_slam_cam
    if tx_base_slam is not None:
        tx_base_cam = tx_base_slam @ tx_slam_cam
    pose_interp = PoseInterpolator(t=timestamp_sec, x=mat_to_pose(tx_base_cam))
    return pose_interp


def get_x_projection(tx_tag_this, tx_tag_other):
    t_this_other = tx_tag_other[:, :3, 3] - tx_tag_this[:, :3, 3]
    v_this_forward = tx_tag_this[:, :3, 2]
    v_up = np.array([0.0, 0.0, 1.0])
    v_this_right = np.cross(v_this_forward, v_up)
    proj_other_right = np.sum(v_this_right * t_this_other, axis=-1)
    return proj_other_right


def slam_to_xarm(width, SLAM_MAX_WIDTH, SLAM_MIN_WIDTH):
    """
    Linearly map a raw (SLAM) gripper width in [0.081, 0.168]
    to xArm control range [-0.01, 0.842].
    Clamps width to [SLAM_MIN_WIDTH, SLAM_MAX_WIDTH].
    """
    XARM_GRIPPER_MIN = -0.01
    XARM_GRIPPER_MAX = 0.800

    w = max(SLAM_MIN_WIDTH, min(SLAM_MAX_WIDTH, width))
    return ((w - SLAM_MIN_WIDTH) / (SLAM_MAX_WIDTH - SLAM_MIN_WIDTH)) * (
        XARM_GRIPPER_MAX - XARM_GRIPPER_MIN
    ) + XARM_GRIPPER_MIN


# -------------------------------------------------------------------------- #
#                                   CLI                                      #
# -------------------------------------------------------------------------- #
@click.command()
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-o', '--output', default=None)
@click.option(
    '-to',
    '--tcp_offset',
    type=float,
    default=0.171,
    help="Distance from gripper tip to mounting screw",
)
@click.option('-ts', '--tx_slam_tag', default=None, help="tx_slam_tag.json")
@click.option(
    '-nz',
    '--nominal_z',
    type=float,
    default=0.072,
    help="nominal Z value for gripper finger tag",
)
@click.option('-ml', '--min_episode_length', type=int, default=24)
@click.option(
    '--ignore_cameras',
    type=str,
    default=None,
    help="comma separated string of camera serials to ignore",
)
def main(
    input,
    output,
    tcp_offset,
    tx_slam_tag,
    nominal_z,
    min_episode_length,
    ignore_cameras,
):
    # ----------------------- stage 0 : paths --------------------------------#
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path / 'demos'
    output = input_path / 'tactile_dataset_plan.pkl'

    # tcp → cam transform (constants)
    cam_to_center_height = 0.082
    cam_to_mount_offset = 0.01465
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset
    pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0, 0, 0])
    tx_cam_tcp = pose_to_mat(pose_cam_tcp)

    # slam ↔ tag transform (optional)
    if tx_slam_tag is None:
        default_path = demos_dir / 'mapping' / 'tx_slam_tag.json'
        tx_slam_tag = str(default_path) if default_path.is_file() else None
    if tx_slam_tag and os.path.isfile(tx_slam_tag):
        tx_slam_tag = np.array(json.load(open(tx_slam_tag, 'r'))['tx_slam_tag'])
        tx_tag_slam = np.linalg.inv(tx_slam_tag)
    else:
        tx_slam_tag = None
        tx_tag_slam = None

    # ------------------- stage 0.b : helpers --------------------------------#
    if ExifToolHelper is not None:
        def meta_reader(p):
            with ExifToolHelper() as et:
                return list(et.get_metadata(str(p)))[0]
    else:
        meta_reader = lambda _: {}  # always empty dict

    # ----------------- load gripper calibration ----------------------------#
    gripper_id_gripper_cal_map = {}
    cam_serial_gripper_cal_map = {}
    for gripper_cal_path in demos_dir.glob('gripper*/gripper_range.json'):
        mp4_path = gripper_cal_path.parent / 'raw_video.mp4'
        if not mp4_path.is_file():
            continue
        meta = meta_reader(mp4_path)
        cam_serial = meta.get('QuickTime:CameraSerialNumber', 'PLACEHOLDER')

        gripper_range_data = json.load(open(gripper_cal_path, 'r'))
        gripper_id = gripper_range_data['gripper_id']
        max_width = gripper_range_data['max_width']
        min_width = gripper_range_data['min_width']
        gripper_cal_data = {
            'aruco_measured_width': [min_width, max_width],
            'aruco_actual_width': [min_width, max_width],
        }
        interp = get_gripper_calibration_interpolator(**gripper_cal_data)
        gripper_id_gripper_cal_map[gripper_id] = interp
        cam_serial_gripper_cal_map[cam_serial] = interp

    # ----------------------- stage 1 : video scan ---------------------------#
    video_dirs = sorted({p.parent for p in demos_dir.glob('demo_*/raw_video.mp4')})
    ignore_cam_serials = set(ignore_cameras.split(',')) if ignore_cameras else set()

    fps = None
    rows = []
    for video_dir in video_dirs:
        mp4_path = video_dir / 'raw_video.mp4'
        orig = video_dir / 'original_raw_video.mp4'
        if not orig.is_file():
            orig = mp4_path

        # camera serial (placeholder-safe)
        try:
            meta = meta_reader(orig)
            cam_serial = meta.get('QuickTime:CameraSerialNumber', 'PLACEHOLDER')
        except Exception:
            cam_serial = 'PLACEHOLDER'

        # tactile required
        if not (video_dir / 'tactile.npy').is_file():
            print(f"[SKIP] {video_dir.name}: tactile.npy missing")
            continue
        if cam_serial in ignore_cam_serials:
            print(f"[SKIP] {video_dir.name}: camera serial ignored")
            continue

        # timestamps (placeholder safe)
        try:
            start_date = mp4_get_start_datetime(str(orig))
            start_ts = start_date.timestamp()
        except Exception:
            start_date, start_ts = None, 0.0

        with av.open(str(mp4_path)) as c:
            st = c.streams.video[0]
            n_frames = st.frames
            local_fps = st.average_rate
            if fps is None:
                fps = local_fps
            elif fps != local_fps:
                print(
                    f"[WARN] fps mismatch {float(fps)} vs {float(local_fps)} "
                    f"in {video_dir.name}"
                )
            duration = float(n_frames / local_fps)
        end_ts = start_ts + duration

        rows.append(
            {
                'video_dir': video_dir,
                'camera_serial': cam_serial,
                'start_date': start_date if start_date else 'PLACEHOLDER',
                'n_frames': n_frames,
                'fps': local_fps,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
            }
        )

    if not rows:
        print("No valid videos found!")
        sys.exit(1)

    video_meta_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------#
    #                     stages 2–5 (unchanged logic)                       #
    # -----------------------------------------------------------------------#
    # … MATCH VIDEOS INTO DEMOS  (code identical to original) …
    serial_count = video_meta_df['camera_serial'].value_counts()
    print("Found cameras:\n", serial_count)
    n_cameras = len(serial_count)

    events = []
    for vid_idx, row in video_meta_df.iterrows():
        events.append(
            dict(
                vid_idx=vid_idx,
                camera_serial=row['camera_serial'],
                t=row['start_timestamp'],
                is_start=True,
            )
        )
        events.append(
            dict(
                vid_idx=vid_idx,
                camera_serial=row['camera_serial'],
                t=row['end_timestamp'],
                is_start=False,
            )
        )
    events.sort(key=lambda x: x['t'])

    demo_data_list, on_videos, on_cams, used_videos = [], set(), set(), set()
    t_demo_start = None
    for evt in events:
        if evt['is_start']:
            on_videos.add(evt['vid_idx'])
            on_cams.add(evt['camera_serial'])
        else:
            on_videos.remove(evt['vid_idx'])
            on_cams.remove(evt['camera_serial'])
        assert len(on_videos) == len(on_cams)

        if len(on_cams) == n_cameras:
            t_demo_start = evt['t']
        elif t_demo_start is not None:
            assert not evt['is_start']
            demo_vids = set(on_videos)
            demo_vids.add(evt['vid_idx'])
            used_videos.update(demo_vids)
            demo_data_list.append(
                dict(
                    video_idxs=sorted(demo_vids),
                    start_timestamp=t_demo_start,
                    end_timestamp=evt['t'],
                )
            )
            t_demo_start = None

    unused = set(video_meta_df.index) - used_videos
    for vid_idx in unused:
        print(f"[WARN] video {video_meta_df.loc[vid_idx, 'video_dir'].name} unused")

    # ----------------------------- stage 3 ----------------------------------#
    # identify gripper id via tag_detection.pkl if present;
    # if absent, use -1 and **continue** (no skipping).
    finger_tag_det_th = 0.8
    vid_idx_gripper_id = {}
    cam_serial_gripper_ids = collections.defaultdict(list)

    for vid_idx, row in video_meta_df.iterrows():
        pkl = row['video_dir'] / 'tag_detection.pkl'
        if not pkl.is_file():
            # no tag file → assume “no gripper” (ID = -1) but keep going
            vid_idx_gripper_id[vid_idx] = -1
        else:
            data = pickle.load(open(pkl, 'rb'))
            n_frames = len(data)
            tag_counts = collections.Counter(
                k for frame in data for k in frame['tag_dict'].keys()
            )
            tag_stats = {k: v / n_frames for k, v in tag_counts.items()}
            if not tag_stats:
                vid_idx_gripper_id[vid_idx] = -1
                continue

            max_tag_id = max(tag_stats)
            tag_per_gripper = 6
            max_gripper_id = max_tag_id // tag_per_gripper
            gripper_prob = {}
            for gid in range(max_gripper_id + 1):
                left, right = gid * tag_per_gripper, gid * tag_per_gripper + 1
                prob = min(tag_stats.get(left, 0), tag_stats.get(right, 0))
                if prob > 0:
                    gripper_prob[gid] = prob
            if not gripper_prob:
                vid_idx_gripper_id[vid_idx] = -1
            else:
                gid, prob = max(gripper_prob.items(), key=lambda kv: kv[1])
                vid_idx_gripper_id[vid_idx] = gid if prob >= finger_tag_det_th else -1

        cam_serial_gripper_ids[row['camera_serial']].append(vid_idx_gripper_id[vid_idx])

    video_meta_df['gripper_hardware_id'] = pd.Series(vid_idx_gripper_id)

    cam_serial_gripper_id_map = {}
    for cs, gids in cam_serial_gripper_ids.items():
        gid = collections.Counter(gids).most_common(1)[0][0]
        cam_serial_gripper_id_map[cs] = gid

    # ----------------------------- stage 4 ----------------------------------#
    n_gripper_cams = (np.array(list(cam_serial_gripper_id_map.values())) >= 0).sum()
    grip_cam_serials = [cs for cs, gi in cam_serial_gripper_id_map.items() if gi >= 0]
    other_cam_serials = [cs for cs, gi in cam_serial_gripper_id_map.items() if gi < 0]

    cam_serial_cam_idx_map = {
        cs: i for i, cs in enumerate(sorted(other_cam_serials), start=len(grip_cam_serials))
    }
    cam_serial_cam_idx_map.update({cs: i for i, cs in enumerate(sorted(grip_cam_serials))})
    
    if 'camera_idx' not in video_meta_df.columns:
        video_meta_df['camera_idx'] = 0
    else:
        video_meta_df['camera_idx'].fillna(0, inplace=True)



    vid_idx_cam_idx_map = np.full(len(video_meta_df), -1, np.int32)
    cam_serial_right_left_idx = collections.defaultdict(list)

    for demo in demo_data_list:
        v_idxs = demo['video_idxs']
        demo_df = video_meta_df.loc[v_idxs].copy()
        demo_df.set_index('camera_idx', inplace=False)

        cam_poses, cam_serials, gripper_vid_idxs, pose_interps = [], [], [], []

        for vid_idx in v_idxs:
            row = video_meta_df.loc[vid_idx]
            cam_serials.append(row['camera_serial'])
            gripper_vid_idxs.append(vid_idx)

            csv = row['video_dir'] / 'camera_trajectory.csv'
            if not csv.is_file():
                continue  # will fallback to placeholder later

            csv_df = pd.read_csv(csv)
            if csv_df['is_lost'].sum() > 10:
                continue
            if (~csv_df['is_lost']).sum() < 60:
                continue

            df = csv_df.loc[~csv_df['is_lost']]
            interp = pose_interp_from_df(
                df,
                start_timestamp=row['start_timestamp'],
                tx_base_slam=tx_tag_slam if tx_tag_slam is not None else None,
            )
            pose_interps.append(interp)

        # For cam ordering we still need some inter-camera geometry; if none
        # exists we default indices deterministically:
        if not pose_interps:
            for vid_idx in v_idxs:
                cs = video_meta_df.loc[vid_idx, 'camera_serial']
                vid_idx_cam_idx_map[vid_idx] = cam_serial_cam_idx_map.get(cs, 0)
            continue

        n_samples = 100
        t0 = demo['start_timestamp']
        t1 = demo['end_timestamp']
        ts = np.linspace(t0, t1, n_samples)
        pose_samples = [pose_to_mat(p(ts)) for p in pose_interps]

        x_proj_avg = []
        for i, p_i in enumerate(pose_samples):
            x_proj_avg.append(
                np.mean([np.mean(get_x_projection(p_i, p_j)) for p_j in pose_samples])
            )
        right_left = np.argsort(x_proj_avg)

        for vid_idx, cs, rl_idx in zip(gripper_vid_idxs, cam_serials, right_left):
            cam_serial_right_left_idx[cs].append(rl_idx)
            vid_idx_cam_idx_map[vid_idx] = rl_idx

    camera_idx_series = video_meta_df['camera_serial'].map(cam_serial_cam_idx_map)
    video_meta_df['camera_idx'] = camera_idx_series
    video_meta_df['camera_idx_from_episode'] = pd.Series(
        vid_idx_cam_idx_map, index=video_meta_df.index
    )

    rows = []
    for cs, ci in cam_serial_cam_idx_map.items():
        rows.append(
            dict(
                camera_idx=ci,
                camera_serial=cs,
                gripper_hw_idx=cam_serial_gripper_id_map.get(cs, -1),
                example_vid=video_meta_df.loc[video_meta_df['camera_serial'] == cs]
                .iloc[0]['video_dir']
                .name,
            )
        )
    camera_serial_df = pd.DataFrame(rows).set_index('camera_idx').sort_index()
    print("Assigned camera_idx: right=0; left=1; non-gripper=2,3…")
    print(camera_serial_df)

    # ----------------------------- stage 6 ----------------------------------#
    total_available, total_used = 0.0, 0.0
    n_dropped_demos = 0
    all_plans = []

    for demo_idx, demo in enumerate(demo_data_list):
        v_idxs = demo['video_idxs']
        s_ts, e_ts = demo['start_timestamp'], demo['end_timestamp']
        total_available += e_ts - s_ts

        demo_df = video_meta_df.loc[v_idxs].copy()
        demo_df.set_index('camera_idx', inplace=True)
        demo_df.sort_index(inplace=True)

        # alignment grid
        dt = None
        costs = []
        for _, row in demo_df.iterrows():
            dt = 1 / row['fps']
            row_cost = []
            for _, o_row in demo_df.iterrows():
                diff = o_row['start_timestamp'] - row['start_timestamp']
                row_cost.append(diff % dt)
            costs.append(row_cost)

        if not costs:
            print(f"[SKIP] demo {demo_idx}: no cameras")
            n_dropped_demos += 1
            continue

        align_idx = np.argmin([sum(c) for c in costs])
        align_start = demo_df.iloc[align_idx]['start_timestamp']
        s_ts += dt - ((s_ts - align_start) % dt)

        cam_start_frames = []
        n_frames = int((e_ts - s_ts) / dt)
        for _, row in demo_df.iterrows():
            vs = math.ceil((s_ts - row['start_timestamp']) / dt)
            vn = math.floor((row['end_timestamp'] - s_ts) / dt) - 1
            if vs < 0:
                vn += vs
                vs = 0
            cam_start_frames.append(vs)
            n_frames = min(n_frames, vn)

        demo_ts = np.arange(n_frames) * float(dt) + s_ts

        cam_poses, cam_w, cam_valid = [], [], []
        for cam_idx, row in demo_df.iterrows():
            start_idx = cam_start_frames[cam_idx]
            v_dir = row['video_dir']

            # trajectory CSV optional
            csv = v_dir / 'camera_trajectory.csv'
            if csv.is_file():
                csv_df = pd.read_csv(csv)
                df = csv_df.iloc[start_idx : start_idx + n_frames]
                is_track = (~df['is_lost']).to_numpy()
                cam_pos = df[['x', 'y', 'z']].fillna(0).to_numpy()
                cam_quat = df[['q_x', 'q_y', 'q_z', 'q_w']].fillna(0).to_numpy()
                rot = Rotation.from_quat(cam_quat)
                tx_tag_tcp = np.zeros((n_frames, 4, 4), np.float32)
                tx_tag_tcp[:, 3, 3] = 1
                for i in range(n_frames):
                    m = np.eye(4, dtype=np.float32)
                    m[:3, 3] = cam_pos[i]
                    m[:3, :3] = rot[i].as_matrix()
                    if tx_tag_slam is not None:
                        tx_tag_tcp[i] = tx_tag_slam @ m @ tx_cam_tcp
                    else:
                        tx_tag_tcp[i] = m @ tx_cam_tcp
                pose_tag_tcp = mat_to_pose(tx_tag_tcp)
            else:
                # no CSV: placeholder
                is_track = np.ones(n_frames, bool)
                pose_tag_tcp = np.full((n_frames, 6), np.nan)

            # tag detection optional
            pkl = v_dir / 'tag_detection.pkl'
            if pkl.is_file():
                td_data = pickle.load(open(pkl, 'rb'))
                td_data = td_data[start_idx : start_idx + n_frames]
            else:
                td_data = [None] * n_frames

            ghi = row['gripper_hardware_id']
            g_width = np.zeros(n_frames, np.float32)
            if ghi >= 0:
                left, right = 6 * ghi, 6 * ghi + 1
                interp = (
                    gripper_id_gripper_cal_map.get(ghi)
                    or cam_serial_gripper_cal_map.get(row['camera_serial'])
                )
                for i, td in enumerate(td_data):
                    if td and interp:
                        width = get_gripper_width(
                            td['tag_dict'], left_id=left, right_id=right, nominal_z=nominal_z
                        )
                        if width is not None:
                            g_width[i] = slam_to_xarm(width, 0.168, 0.081)

            cam_poses.append(pose_tag_tcp)
            cam_w.append(g_width)
            cam_valid.append(is_track)

        if not cam_poses:
            print(f"[SKIP] demo {demo_idx}: no valid cameras after placeholder processing")
            n_dropped_demos += 1
            continue

        cam_valid = np.array(cam_valid)
        is_step_valid = np.all(cam_valid, axis=0)
        if not is_step_valid.any():
            print(f"[SKIP] demo {demo_idx}: zero valid frames")
            n_dropped_demos += 1
            continue

        first, last = np.nonzero(is_step_valid)[0][[0, -1]]

        segs, seg_type = get_bool_segments(is_step_valid)
        for s, valid in zip(segs, seg_type):
            if not valid:
                continue
            if (s.stop - s.start) < min_episode_length:
                is_step_valid[s] = False

        segs, seg_type = get_bool_segments(is_step_valid)
        for s, valid in zip(segs, seg_type):
            if not valid:
                continue
            total_used += (s.stop - s.start) * dt

            cameras, grippers = [], []
            for idx_cam, row in demo_df.iterrows():
                v_dir = row['video_dir']
                vs = cam_start_frames[idx_cam]
                cameras.append(
                    dict(
                        video_path=str((v_dir / 'raw_video.mp4').relative_to(v_dir.parent)),
                        video_start_end=(s.start + vs, s.stop + vs),
                    )
                )

            for cidx in range(len(cam_poses)):
                pose_tcp = cam_poses[cidx][s.start : s.stop]
                g_width = cam_w[cidx][s.start : s.stop]
                grippers.append(
                    dict(
                        tcp_pose=pose_tcp,
                        gripper_width=g_width,
                        demo_start_pose=pose_tcp[0] if len(pose_tcp) > 0 else np.full(6, np.nan),
                        demo_end_pose=pose_tcp[-1] if len(pose_tcp) > 0 else np.full(6, np.nan),
                    )
                )

            all_plans.append(
                dict(episode_timestamps=demo_ts[s.start : s.stop], grippers=grippers, cameras=cameras)
            )

    used_ratio = total_used / (total_available + 1e-9)
    print(f"{used_ratio*100:.1f}% of raw data kept")
    print("n_dropped_demos:", n_dropped_demos)

    with open(output, 'wb') as f:
        pickle.dump(all_plans, f)
    print(f"Dataset plan saved → {output}")


def test():
    pass


if __name__ == "__main__":
    main()
