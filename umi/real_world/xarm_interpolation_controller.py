import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import scipy.spatial.transform as st

# Make sure these match your project imports
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty
)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer
)
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from diffusion_policy.common.precise_sleep import precise_wait

from xarm.wrapper import XArmAPI

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class XArmInterpolationController(mp.Process):
    """
    xArm servo controller aligned with UR5 & Franka style.
    We do NOT disconnect immediately after reading example data in the constructor.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        frequency=125,
        lookahead_time=0.1,   # UR/Franka param (unused in xArm servo calls)
        gain=300,            # UR/Franka param (unused in xArm servo calls)
        max_pos_speed=0.1,
        max_rot_speed=0.16,
        launch_timeout=3,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=1.0,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=2500,
        receive_latency=0.0
    ):
        super().__init__(name="XArmPositionalController")

        if joints_init is not None:
            joints_init = np.array(joints_init, dtype=float)
            assert joints_init.shape[0] >= 6

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # Matching UR5 approach: keep a receive_latency variable
        self.receive_latency = receive_latency

        # -------------------------------
        # 1) Create an xArm connection ONCE, then read initial data for ring buffer
        self.arm = XArmAPI(self.robot_ip)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        # Switch to position mode to read states
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(1)

        # Acquire real data for ring buffer example
        code, actual_pose_aa = self.arm.get_position_aa(is_radian=True)
        if code != 0 or actual_pose_aa is None:
            actual_pose_aa = np.zeros((6,), dtype=np.float64)
            if self.verbose:
                print("[XArmPositionalController] Warning: failed to get initial pose from xArm.")

        code, actual_q = self.arm.get_servo_angle(is_radian=True)
        if code != 0 or actual_q is None:
            actual_q = np.zeros((7,), dtype=np.float64)
        else:
            if len(actual_q) < 7:
                padded_q = np.zeros(7, dtype=np.float64)
                padded_q[:len(actual_q)] = actual_q
                actual_q = padded_q

        # Build the ring buffer example
        example = {}
        example['ActualTCPPose']   = np.array(actual_pose_aa)
        example['ActualTCPSpeed']  = np.zeros(6, dtype=np.float64)
        example['ActualQ']         = np.array(actual_q)
        example['ActualQd']        = np.zeros(7, dtype=np.float64)

        example['TargetTCPPose']   = np.zeros(6, dtype=np.float64)
        example['TargetTCPSpeed']  = np.zeros(6, dtype=np.float64)
        example['TargetQ']         = np.zeros(7, dtype=np.float64)
        example['TargetQd']        = np.zeros(7, dtype=np.float64)

        # Notice: We keep both timestamps to mimic UR5 style
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()

        # 2) Build input queue
        example_cmd = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=256
        )

        # 3) Possibly define the receive_keys
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose', 'ActualTCPSpeed',
                'ActualQ', 'ActualQd',
                'TargetTCPPose', 'TargetTCPSpeed',
                'TargetQ', 'TargetQd'
            ]
        self.receive_keys = receive_keys

        # 4) Create ring buffer
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()

    # Lifecycle
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[XArmPositionalController] started (pid={self.pid})")

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # Commands
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert duration >= (1. / self.frequency)
        pose = np.array(pose, dtype=np.float64)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert self.is_alive()
        pose = np.array(pose, dtype=np.float64)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        # enable soft real-time if requested
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except PermissionError:
                if self.verbose:
                    print("[XArmPositionalController] no permission for SCHED_RR")

        # Connect to xArm and initialize
        arm = XArmAPI(self.robot_ip)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.clean_error()

        # Normal mode first
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)


        # Switch to servo mode
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(1) 


        # Move to joints_init if given
        if self.joints_init is not None:
            code = arm.set_servo_angle(angle=self.joints_init, is_radian=True)
            if code != 0 and self.verbose:
                print(f"[XArmPositionalController] Warning: init joints failed, code={code}")
            time.sleep(self.joints_init_speed)


        dt = 1.0 / self.frequency
        # Get initial pose for the trajectory
        code, init_pos = arm.get_position_aa(is_radian=True)
        curr_t = time.monotonic()
        last_waypoint_time = curr_t

        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[init_pos]
        )

        
        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        try:
            while keep_running:
                # Mark the start time of current iteration (for freq measurement)
                t_loop_start = time.perf_counter()
                t_now = time.monotonic()

                # Evaluate interpolated pose
                target_pose_aa = pose_interp(t_now)

                # Send servo command
                ret = arm.set_servo_cartesian_aa(
                    target_pose_aa,
                    is_radian=True,
                    relative=False
                )


                # Collect actual state and set timestamps
                state = self._get_robot_state(arm, target_pose_aa)
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # Process commands (matching UR5 approach with get_k(1))
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = commands['target_pose'][i]
                        duration = float(commands['duration'][i])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(f"[XArmPositionalController] servoL to {target_pose}, duration={duration}")
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = commands['target_pose'][i]
                        global_target_time = float(commands['target_time'][i])
                        monotonic_target_time = (time.monotonic() - time.time()) + global_target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=monotonic_target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = monotonic_target_time
                    else:
                        keep_running = False
                        break

                # Use precise_wait to maintain frequency (matching UR5 code)
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop done => set ready_event
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    freq_est = 1.0 / (time.perf_counter() - t_loop_start)
                    print(f"[XArmPositionalController] freq ~ {freq_est:.1f} Hz")

        finally:
            # Cleanup
            arm.set_state(4)  # Stopped
            arm.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[XArmPositionalController] disconnected from {self.robot_ip}")


    def _get_robot_state(self, arm: XArmAPI, target_pose_aa):
        # Actual pose
        code, actual_pose_aa = arm.get_position_aa(is_radian=True)
        if code != 0 or actual_pose_aa is None:
            actual_pose_aa = np.zeros(6, dtype=np.float64)

        #actual_pose_aa[:3] = actual_pose_aa[:3]/1000

        # Actual joints
        code, actual_q_raw = arm.get_servo_angle(is_radian=True)
        if code != 0 or actual_q_raw is None:
            actual_q_raw = []
        actual_q = np.array(actual_q_raw, dtype=np.float64)
        if actual_q.shape[0] < 7:
            pad = np.zeros(7, dtype=np.float64)
            pad[:actual_q.shape[0]] = actual_q
            actual_q = pad

        # We do not have real-time measured velocities from xArm, so we store zeros
        ActualTCPSpeed = np.zeros(6, dtype=np.float64)
        ActualQd = np.zeros(7, dtype=np.float64)

        # For “target” fields, store the last commanded pose in “TargetTCPPose”
        TargetTCPPose = target_pose_aa
        TargetTCPSpeed = np.zeros(6, dtype=np.float64)
        TargetQ = np.zeros(7, dtype=np.float64)
        TargetQd = np.zeros(7, dtype=np.float64)

        return {
            'ActualTCPPose': actual_pose_aa,
            'ActualTCPSpeed': ActualTCPSpeed,
            'ActualQ': actual_q,
            'ActualQd': ActualQd,
            'TargetTCPPose': TargetTCPPose,
            'TargetTCPSpeed': TargetTCPSpeed,
            'TargetQ': TargetQ,
            'TargetQd': TargetQd,
            # timestamps are set in the main loop
        }
