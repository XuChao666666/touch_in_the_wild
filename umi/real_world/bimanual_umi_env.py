from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.xarm_gripper_controller import *
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.xarm_interpolation_controller import XArmInterpolationController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import draw_predefined_mask
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator


class BimanualUmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            robots_config, # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
            grippers_config, # list of dict[{gripper_ip: XXX, gripper_port: 1000, obs_latency: 0.01, , action_latency: 0.1}]
            # env params
            frequency=20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=80,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=True,
            fisheye_converter=None,
            mirror_swap=False,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            init_joints=False,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            tactile_controller_left=None,
            tactile_controller_right=None,
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        self.tactile_data = None

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug.
        reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths

        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for path in v4l_paths:
            if 'Cam_Link_4K' in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000*1000

                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=False, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)
                
                def tf(data, input_res=res):
                    img = data['color']
                    if fisheye_converter is None:
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution, 
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0), 
                            mirror=False, gripper=False, finger=False, use_aa=True)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data, input_res=res):
                img = data['color']
                f = get_image_transform(
                    input_res=input_res,
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None

        assert len(robots_config) == len(grippers_config)
        robots: List[RTDEInterpolationController] = list()
        grippers: List[WSGController] = list()
        for rc in robots_config:
            if rc['robot_type'].startswith('ur5'):
                assert rc['robot_type'] in ['ur5', 'ur5e']
                this_robot = RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=500 if rc['robot_type'] == 'ur5e' else 125,
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=3,
                    tcp_offset_pose=[0, 0, rc['tcp_offset'], 0, 0, 0],
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=j_init,
                    joints_init_speed=1.05,
                    soft_real_time=False,
                    verbose=False,
                    receive_keys=None,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc['robot_type'].startswith('franka'):
                this_robot = FrankaInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=200,
                    Kx_scale=1.0,
                    Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                    verbose=False,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc['robot_type'].startswith('xarm'):
                # Minimal xArm block
                this_robot = XArmInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=200,    # or rc.get('frequency', 125) if you want to pick it from the config
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=80,
                    max_rot_speed=1.57,
                    launch_timeout=3,
                    tcp_offset_pose=None,       # or pass rc['tcp_offset'] if you want
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=None,         # Not added
                    joints_init_speed=1.0,
                    soft_real_time=False,
                    verbose=False,
                    receive_latency=rc['robot_obs_latency']
                )
            else:
                raise NotImplementedError()
            robots.append(this_robot)

        
        for rc, gc in zip(robots_config, grippers_config):
            if rc['robot_type'].startswith('xarm'):
                # Use the XArmGripperController for xArm robots
                this_gripper = XArmGripperController(
                    shm_manager=shm_manager,
                    hostname=gc['gripper_ip'],
                    port=gc['gripper_port'],
                    receive_latency=0.001
                )
            else:
                # For UR/Franka, use WSGController as before
                this_gripper = WSGController(
                    shm_manager=shm_manager,
                    hostname=gc['gripper_ip'],
                    port=gc['gripper_port'],
                    receive_latency=0.001,
                    use_meters=True
                )

            grippers.append(this_gripper)


        '''
        for gc in grippers_config:
            this_gripper = WSGController(
                shm_manager=shm_manager,
                hostname=gc['gripper_ip'],
                port=gc['gripper_port'],
                receive_latency=gc['gripper_obs_latency'],
                use_meters=True
            )

            grippers.append(this_gripper)
        '''

        self.camera = camera
        
        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config

        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.tactile_left = tactile_controller_left
        self.tactile_right = tactile_controller_right
        self.tactile_obs_horizon = camera_obs_horizon
        self.tactile_down_sample_steps = camera_down_sample_steps
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self.last_time_step = 0
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
        return ready_flag
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        # -- [A] Measure how often we call get_obs() in wall-clock time --
        if not hasattr(self, '_last_obs_walltime'):
            self._last_obs_walltime = time.time()
            print("[DEBUG] First call to get_obs(), no previous timing.")
        else:
            curr_time = time.time()
            dt_wall = curr_time - self._last_obs_walltime
            self._last_obs_walltime = curr_time
            #print(f"[DEBUG] Time since last get_obs() call = {dt_wall:.3f} s (~{1.0/dt_wall:.1f} Hz)")

        # Decide how many frames we want from each camera
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency)
        ) + 2
        # For debugging:
        # print(f"[DEBUG] We'll fetch up to k={k} frames from each camera.")

        # -- [B] Grab camera data and measure their timestamps --
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)

        # Print how many frames we got from each camera and approximate frequency
        for cam_idx, cam_dict in self.last_camera_data.items():
            cam_timestamps = cam_dict['timestamp']  # shape (k,)
            if cam_timestamps.shape[0] > 1:
                deltas = cam_timestamps[1:] - cam_timestamps[:-1]
                avg_delta = np.mean(deltas)
                freq = 1.0 / avg_delta if avg_delta > 1e-8 else 9999.0
                #print(f"[DEBUG] Camera {cam_idx}: pulled {len(cam_timestamps)} frames. "
                #    f"Avg dt={avg_delta:.4f}s -> ~{freq:.1f} Hz.")
            else:
                print(f"[DEBUG] Camera {cam_idx}: only {cam_timestamps.shape[0]} frame(s).")

        # -- [C] Pull all robot & gripper states from ring buffers --
        last_robots_data = []
        for robot_idx, robot in enumerate(self.robots):
            rd = robot.get_all_state()  # ring buffer data
            last_robots_data.append(rd)
            # Quick measure of how often new states arrived in that buffer
            r_ts = rd['robot_timestamp']

            # Print the "robot world" current time
            #if r_ts.shape[0] > 0:
            #    print(f"[DEBUG] Robot{robot_idx} 'robot world' current time: {r_ts[-1]:.3f}")

        last_grippers_data = []
        for gripper_idx, gripper in enumerate(self.grippers):
            gd = gripper.get_all_state()
            last_grippers_data.append(gd)
            g_ts = gd['gripper_timestamp']
            if g_ts.shape[0] > 1:
                deltas = g_ts[1:] - g_ts[:-1]
                avg_dt = np.mean(deltas)
                freq = (1.0 / avg_dt) if avg_dt > 0 else 9999
                #print(f"[DEBUG] Gripper{gripper_idx} ring buffer: {g_ts.shape[0]} states, "
                #    f"avg dt={avg_dt:.4f}s (~{freq:.1f} Hz).")
            else:
                print(f"[DEBUG] Gripper{gripper_idx} ring buffer: {g_ts.shape[0]} state(s).")


        # ===== Camera alignment logic (unchanged) =====
        num_obs_cameras = len(self.robots)
        align_camera_idx = None
        running_best_error = np.inf

        for camera_idx in range(num_obs_cameras):
            this_error = 0.0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        # The 'align_camera_idx' camera's latest frame sets the "last_timestamp"
        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # ===== [D] Build camera_obs_timestamps ======
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt
        )

        # Let's also print them in decimal form (instead of scientific notation):
        camera_obs_str = [f"{ts:.6f}" for ts in camera_obs_timestamps]
        '''
        if len(camera_obs_timestamps) >= 2:
            print(f"[DEBUG] camera_obs_timestamps (last 2):"
                f" {camera_obs_timestamps[-2]:.4f}, {camera_obs_timestamps[-1]:.4f}")
            print(f"[DEBUG] camera time delta ~ {camera_obs_timestamps[-1] - camera_obs_timestamps[-2]:.4f} s")
        else:
            print(f"[DEBUG] camera_obs_timestamps: {camera_obs_timestamps.tolist()}")
        '''



        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = []
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        obs_data = dict(camera_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        # ===== [E] Align robot observations =====
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt
        )

        robot_obs_str = [f"{ts:.6f}" for ts in robot_obs_timestamps]
        #print("[DEBUG] robot_obs_timestamps:", robot_obs_str)
        if len(robot_obs_timestamps) > 1:
            dt_robot = robot_obs_timestamps[1] - robot_obs_timestamps[0]
            #print(f"[DEBUG] => The {len(robot_obs_timestamps)} robot observation frames are "
            #    f"{dt_robot:.3f}s apart (~{1.0/dt_robot:.1f} Hz).")

        for robot_idx, last_robot_data in enumerate(last_robots_data):
            # Interpolate poses
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'],
                x=last_robot_data['ActualTCPPose']
            )
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)  # shape (N,6)
            obs_data[f'robot{robot_idx}_eef_pos'] = robot_pose[..., :3]
            obs_data[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[..., 3:]

        # ===== [F] Align gripper observations =====
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt
        )

        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][..., None]
            )

            # 1) Interpolate now to get the gripper widths at the requested timestamps:
            gripper_width_vals = gripper_interpolator(gripper_obs_timestamps)
            obs_data[f'robot{robot_idx}_gripper_width'] = gripper_width_vals

            # 2) Now we can safely reference `gripper_width_vals` in debug prints:
            if self.gripper_obs_horizon >= 2:
                # For example, print the last 2 timestamps & values:
                last_2_times = gripper_obs_timestamps[-2:]
                last_2_values = gripper_width_vals[-2:, 0]  # shape (N,1), pick index 0 in last axis
                print(f"[DEBUG] Gripper{robot_idx} last 2 requested timesteps:"
                    f" { [f'{t:.6f}' for t in last_2_times] }")
                print(f"[DEBUG] Corresponding gripper_width values = {last_2_values.tolist()}")
    


        # ===== [G] Optionally accumulate to obs_accumulator if desired =====
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )
            for robot_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][..., None]
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )
        
        # --- Now Tactile logic ---
        if (self.tactile_left is not None) and (self.tactile_right is not None):
            k_tactile = int(np.ceil(
                self.tactile_obs_horizon
                * self.tactile_down_sample_steps
                * (60 / self.frequency)
            )) + 2

            data_left = self.tactile_left.get(k=k_tactile)
            data_right = self.tactile_right.get(k=k_tactile)

            if len(data_left['timestamp']) > 0:
                left_last_ts = data_left['timestamp'][-1]
            else:
                left_last_ts = time.time()

            if len(data_right['timestamp']) > 0:
                right_last_ts = data_right['timestamp'][-1]
            else:
                right_last_ts = time.time()

            last_tactile_ts = max(left_last_ts, right_last_ts)

            tactile_obs_timestamps = last_tactile_ts - (
                np.arange(self.tactile_obs_horizon)[::-1]
                * self.tactile_down_sample_steps
                * dt
            )

            '''
            if len(tactile_obs_timestamps) >= 2:
                print(f"[DEBUG] tactile_obs_timestamps (last 2):"
                    f" {tactile_obs_timestamps[-2]:.4f}, {tactile_obs_timestamps[-1]:.4f}")
                print(f"[DEBUG] tactile time delta ~ "
                    f"{tactile_obs_timestamps[-1] - tactile_obs_timestamps[-2]:.4f} s")
            else:
                print(f"[DEBUG] tactile_obs_timestamps: {tactile_obs_timestamps.tolist()}")
            '''



            ring_ts_left = data_left['timestamp']
            ring_fr_left = data_left['frame']
            ring_ts_right = data_right['timestamp']
            ring_fr_right = data_right['frame']

            left_frames = []
            right_frames = []
            for desired_t in tactile_obs_timestamps:
                if len(ring_ts_left) < 1:
                    left_frames.append(np.zeros((12,32), dtype=np.float32))
                else:
                    idx_l = np.argmin(np.abs(ring_ts_left - desired_t))
                    left_frames.append(ring_fr_left[idx_l])

                if len(ring_ts_right) < 1:
                    right_frames.append(np.zeros((12,32), dtype=np.float32))
                else:
                    idx_r = np.argmin(np.abs(ring_ts_right - desired_t))
                    right_frames.append(ring_fr_right[idx_r])

            obs_data['tactile_left']  = np.stack(left_frames, axis=0)
            obs_data['tactile_right'] = np.stack(right_frames, axis=0)

            # Optionally create camera0_tactile by combining left & right horizontally
            combined_lr = []
            for lf, rf in zip(left_frames, right_frames):
                combined_lr.append(np.concatenate([lf, rf], axis=-1))
            combined_lr = np.stack(combined_lr, axis=0)  # shape (H,12,64)
            combined_lr = combined_lr[None]              # shape (1,H,12,64)
            obs_data['camera0_tactile'] = combined_lr

            if self.tactile_obs_horizon >= 2:
                last_2_tactile_left_times = []
                last_2_tactile_right_times = []
                for desired_t in tactile_obs_timestamps[-2:]:
                    idx_l = np.argmin(np.abs(ring_ts_left - desired_t))
                    idx_r = np.argmin(np.abs(ring_ts_right - desired_t))
                    left_time  = ring_ts_left[idx_l]  if len(ring_ts_left) > 0  else None
                    right_time = ring_ts_right[idx_r] if len(ring_ts_right) > 0 else None
                    last_2_tactile_left_times.append(left_time)
                    last_2_tactile_right_times.append(right_time)


            #print(f"[DEBUG] last_2 ring buffer times (left): {last_2_tactile_left_times}")
            #print(f"[DEBUG] last_2 ring buffer times (right): {last_2_tactile_right_times}")


        else:
            obs_data['tactile_left'] = None
            obs_data['tactile_right'] = None
            obs_data['camera0_tactile'] = None

        if obs_data['tactile_left'] is not None:
            cam_t = obs_data['timestamp'][-1]          # camera timeline, last frame
            tac_t = tactile_obs_timestamps[-1]         # tactile timeline, last frame
            print(f"[Δ camera-tactile] {cam_t - tac_t:+.3f}  sec")





        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        assert new_actions.shape[1] // len(self.robots) == 7
        assert new_actions.shape[1] % len(self.robots) == 0

        # schedule waypoints
        for i in range(len(new_actions)):
            for robot_idx, (robot, gripper, rc, gc) in enumerate(zip(self.robots, self.grippers, self.robots_config, self.grippers_config)):
                r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                g_actions = new_actions[i, 7 * robot_idx + 6]
                robot.schedule_waypoint(
                    pose=r_actions,
                    target_time=new_timestamps[i] - r_latency
                )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    # recording API
    '''
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    '''

    def start_episode(self, start_time=None):
        """
        Start recording:
        1) Original camera videos (one per camera).
        2) A second 'merged.mp4' containing camera + tactile in a single frame.
        """
        import cv2

        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        # Create folder for this episode's videos
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)

        # (A) Original camera MP4(s):
        n_cameras = self.camera.n_cameras
        video_paths = []
        for i in range(n_cameras):
            out_path = this_video_dir.joinpath(f'{i}.mp4').absolute()
            video_paths.append(str(out_path))

        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # (B) Create a cv2.VideoWriter for merged output
        merged_path = str(this_video_dir.joinpath('merged.mp4').absolute())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # use H.264 for better clarity

        # We'll set the size to "None" for now; we will create the writer
        # dynamically once we know the exact final_img shape the first time we write.
        self.merged_video_writer = None
        self.merged_video_path = merged_path
        self.merged_video_fourcc = fourcc

        # (C) Prepare a tactile-only writer
        tactile_path = str(this_video_dir.joinpath('tactile.mp4').absolute())
        self.tactile_video_writer = None
        self.tactile_video_path = tactile_path
        self.tactile_video_fourcc = fourcc
        self.tactile_fps = self.frequency

        # (D) Create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/60
        )
        print(f"[BimanualUmiEnv] Episode {episode_id} started recording.")

    def end_episode(self):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.camera.stop_recording()

        if getattr(self, 'merged_video_writer', None) is not None:
            self.merged_video_writer.release()
            self.merged_video_writer = None

        # 3) If you had a separate "tactile_video_writer", close that too.
        if getattr(self, 'tactile_video_writer', None) is not None:
            self.tactile_video_writer.release()
            self.tactile_video_writer = None

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                    episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                    )
                    episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                    episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)

                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_gripper_width']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_gripper_width'])
                    )
                    episode[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
