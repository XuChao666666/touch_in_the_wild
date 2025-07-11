"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
import time

import matplotlib.pyplot as plt  ### ADDED FOR PLOTTING ###
from mpl_toolkits.mplot3d import Axes3D  ### ADDED FOR PLOTTING ###

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.tactile_controller_left import TactileControllerLeft
from umi.real_world.tactile_controller_right import TactileControllerRight
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

def write_merged_frame(env, frame, arr_left, arr_right):
    """
    Merges the camera 'frame' on top with the tactile side-by-side on bottom,
    then writes to env.merged_video_writer (creating if needed).
    """
    import cv2
    import numpy as np

    left_vis_u8  = (arr_left  * 255).clip(0,255).astype(np.uint8)
    right_vis_u8 = (arr_right * 255).clip(0,255).astype(np.uint8)
    left_color  = cv2.applyColorMap(left_vis_u8,  cv2.COLORMAP_VIRIDIS)
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

    pad_w = 10  # pixels of padding between left and right tactile
    pad = np.zeros((left_color_big.shape[0], pad_w, 3), dtype=np.uint8)
    tactile_vis = np.hstack((left_color_big, pad, right_color_big))

    if env.tactile_video_writer is None:
        h_t, w_t = tactile_vis.shape[:2]
        env.tactile_video_writer = cv2.VideoWriter(
            env.tactile_video_path,
            env.tactile_video_fourcc,
            env.tactile_fps,
            (w_t, h_t),
            True
        )
    env.tactile_video_writer.write(tactile_vis.astype(np.uint8))

    tvis_h, tvis_w = tactile_vis.shape[:2]
    cam_h, cam_w = frame.shape[:2]

    new_cam_h = int(cam_h * (tvis_w / float(cam_w)))
    cam_resized = cv2.resize(frame, (tvis_w, new_cam_h), interpolation=cv2.INTER_AREA)

    final_img = np.vstack((cam_resized, tactile_vis))

    if env.merged_video_writer is None:
        out_h, out_w = final_img.shape[:2]
        env.merged_video_writer = cv2.VideoWriter(
            env.merged_video_path,
            env.merged_video_fourcc,
            20,
            (out_w, out_h),
            True
        )

    final_img_8u = final_img.astype(np.uint8)
    env.merged_video_writer.write(final_img_8u)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=True)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    import numpy as np
    max_gripper_width = 0.9
    gripper_speed = 1000
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    import torch
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        tactile_left = TactileControllerLeft(
            shm_manager=shm_manager,
            port_left="/dev/left_finger_real",
            baud=2000000,
            median_samples=30,
            ring_buffer_size=1000,
            receive_latency=0.06
        )
        # TactileControllerRight
        tactile_right = TactileControllerRight(
            shm_manager=shm_manager,
            port_right="/dev/right_finger",
            baud=2000000,
            median_samples=30,
            ring_buffer_size=1000,
            receive_latency=0.06
        )

        # Start each controller in background threads
        tactile_left.start()
        tactile_right.start()
        print("[Main] Left and Right tactile started in background threads.")


        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=True,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=1.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager,
                tactile_controller_left=tactile_left,
                tactile_controller_right=tactile_right) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # get rid of one dimension for tactile so that the shape is torch.Size([1, 2, 12, 64])
                obs_dict["camera0_tactile"] = obs_dict["camera0_tactile"].squeeze(1)
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    
                    # match shapes                
                    h1, w1, _ = pkl_first_frame.shape
                    h2, w2, _ = vis_img.shape

                    f = get_image_transform((w1, h1), (w2, h2), bgr_to_rgb=False)
                    match_img = f(pkl_first_frame).astype(np.float32)
                    vis_img = (vis_img + match_img)/2
                    #print("finished averaging")
                    
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()

                    if (obs['tactile_left'] is not None) and (obs['tactile_right'] is not None):
                        left_tactile = obs['tactile_left'][-1]    # shape (12,32)
                        right_tactile = obs['tactile_right'][-1]  # shape (12,32)

                        # Convert to [0..255] for colormap
                        left_u8  = (left_tactile  * 255.0).clip(0,255).astype(np.uint8)
                        right_u8 = (right_tactile * 255.0).clip(0,255).astype(np.uint8)

                        left_mapped  = cv2.applyColorMap(left_u8,  cv2.COLORMAP_VIRIDIS)  # (12,32,3)
                        right_mapped = cv2.applyColorMap(right_u8, cv2.COLORMAP_VIRIDIS)  # (12,32,3)

                        tactile_mapped = np.concatenate([left_mapped, right_mapped], axis=1)  # (12,64,3)

                        desired_height = 240
                        scale = desired_height / tactile_mapped.shape[0]
                        tactile_resized = cv2.resize(
                            tactile_mapped,
                            (0,0),
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_NEAREST
                        )
                        cv2.imshow('tactile_view', tactile_resized[..., ::-1])
                    else:
                        blank_tactile = np.full((240, 480, 3), 128, dtype=np.uint8)  # 240x480 gray
                        cv2.putText(blank_tactile, "No Tactile", (10,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                        cv2.imshow('tactile_view', blank_tactile)

                                        
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)
                    
                    

                    

                    action = np.zeros((7 * target_pose.shape[0],))
                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]

                    print("target action",action)

                    
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    obs = env.get_obs()
                    # Important!!!!
                    obs['robot0_eef_pos'][:3] /= 1000.0

                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        obs = env.get_obs()
                        obs['robot0_eef_pos'][:3] /= 1000.0

                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            obs_dict["camera0_tactile"] = obs_dict["camera0_tactile"].squeeze(1)
            
                            result = policy.predict_action(obs_dict)
                            
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            obs['robot0_eef_pos'][:3] *= 1000.0
                            raw_action[:,:3] *= 1000
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            
                            print('Inference latency:', time.time() - s)
                        
                        this_target_poses = action
                        assert this_target_poses.shape[1] == len(robots_config) * 7

                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    height_threshold=robots_config[robot_idx]['height_threshold']
                                )
                            
                            # solve collison between two robots
                            solve_sphere_collision(
                                ee_poses=target_pose.reshape([len(robots_config), -1]),
                                robots_config=robots_config
                            )


                        print("this target poses", this_target_poses)
                       

                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        cam_frame_rgb = obs['camera0_rgb'][-1]  # shape (H,W,3), in [0..1], but channels=RGB
                        cam_frame_bgr = cam_frame_rgb[..., ::-1]
                        cam_frame_u8   = (cam_frame_bgr * 255.0).clip(0,255).astype(np.uint8)
                        arr_left  = obs['tactile_left'][-1]  # shape (12,32) float in [0..1]
                        arr_right = obs['tactile_right'][-1] # shape (12,32) float in [0..1]s
                        # Write to merged:
                        write_merged_frame(env, cam_frame_u8, arr_left, arr_right)
                        
                        

                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            env.replay_buffer.n_episodes, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()

                        if (obs['tactile_left'] is not None) and (obs['tactile_right'] is not None):
                            left_tactile = obs['tactile_left'][-1]    # shape (12,32)
                            right_tactile = obs['tactile_right'][-1]  # shape (12,32)

                            # Convert to [0..255] for colormap
                            left_u8  = (left_tactile  * 255.0).clip(0,255).astype(np.uint8)
                            right_u8 = (right_tactile * 255.0).clip(0,255).astype(np.uint8)

                            left_mapped  = cv2.applyColorMap(left_u8,  cv2.COLORMAP_VIRIDIS)  # (12,32,3)
                            right_mapped = cv2.applyColorMap(right_u8, cv2.COLORMAP_VIRIDIS)  # (12,32,3)

                            tactile_mapped = np.concatenate([left_mapped, right_mapped], axis=1)  # (12,64,3)

                            desired_height = 240
                            scale = desired_height / tactile_mapped.shape[0]
                            tactile_resized = cv2.resize(
                                tactile_mapped,
                                (0,0),
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_NEAREST
                            )
                            cv2.imshow('tactile_view', tactile_resized[..., ::-1])
                        else:
                            blank_tactile = np.full((240, 480, 3), 128, dtype=np.uint8)  # 240x480 gray
                            cv2.putText(blank_tactile, "No Tactile", (10,50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                            cv2.imshow('tactile_view', blank_tactile)
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            tactile_left.stop()
                            tactile_right.stop()
                            break

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    tactile.stop()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
