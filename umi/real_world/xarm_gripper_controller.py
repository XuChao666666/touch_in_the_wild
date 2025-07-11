import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from umi.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

from xarm.wrapper import XArmAPI

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class XArmGripperController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 hostname,
                 port=1000,
                 frequency=40,
                 home_to_open=True,
                 move_max_speed=5000.0,
                 get_max_k=None,
                 command_queue_size=1024,
                 launch_timeout=3,
                 receive_latency=0.0,
                 verbose=False
                 ):
        """
        This is minimally edited from WSGController:
          - permanent scale = 10000.0
          - xArmAPI for the actual control
          - no WSG homing/fault clearance, replaced by xArm calls
        Everything else is the same structure.
        """
        super().__init__(name="XArmGripperController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0  # permanent scale for xArm gripper
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[XArmGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.SHUTDOWN.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # Instead of WSGBinaryDriver, we use xArmAPI here
        arm = XArmAPI(self.hostname)
        try:
            arm.connect()
            arm.motion_enable(True)
            arm.clean_error()
            # The normal approach for xArm gripper initialization:
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(1)

            # set gripper mode & enable
            code = arm.set_gripper_mode(0)
            if code != 0:
                raise RuntimeError(f"Failed to set xArm gripper mode, code={code}")
            code = arm.set_gripper_enable(True)
            if code != 0:
                raise RuntimeError(f"Failed to enable xArm gripper, code={code}")

            # set gripper speed
            code = arm.set_gripper_speed(self.move_max_speed)
            if code != 0:
                raise RuntimeError(f"Failed to set xArm gripper speed, code={code}")

            # read initial position
            code, gpos = arm.get_gripper_position()
            if code != 0 or gpos is None:
                gpos = 0.0
            # print('initial gripper pose', gpos)
            # Scale from mm to local coordinate
            init_pos = gpos / self.scale
            print('initial gripper pose', init_pos)

            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[init_pos,0,0,0,0,0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]

                # command xArm gripper in mm
                # print("target pos", target_pos)
                # print("self.scale", self.scale)
                gripper_pos = target_pos * self.scale
                # print("gripper_pos", gripper_pos)
                code = arm.set_gripper_position(gripper_pos, wait=False)

                # read back current pos
                code, curr_gpos = arm.get_gripper_position()
                if code != 0 or curr_gpos is None:
                    curr_gpos = target_pos * self.scale
                curr_gpos /= self.scale

                # put into ring buffer
                state = {
                    'gripper_state': 0,  # xArm does not provide direct 'state'
                    'gripper_position': curr_gpos,
                    'gripper_velocity': 0.0,    # xArm gripper has no direct velocity
                    'gripper_force': 0.0,       # xArm gripper has no direct force
                    'gripper_measure_timestamp': time.time(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # scale user pos from local range to mm
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # convert global time to monotonic
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos,0,0,0,0,0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            arm.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[XArmGripperController] Disconnected from robot: {self.hostname}")
