import os
import time
import serial
import threading
import numpy as np
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

THRESHOLD   = 10
NOISE_SCALE = 40

def temporal_filter(new_frame, prev_frame, alpha=0.5):
    """
    Temporal smoothing filter. Higher alpha => more weight on new_frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame


def readThreadLeft(serDev, ring_buffer, median_frames, is_running, receive_latency):
    """
    Reads the left tactile sensor lines from `serDev` in 16-line blocks
    and writes the final (12x32) float array to `ring_buffer`.
    Skips any block that does not have the expected shape.
    Also prints the frequency every 2 valid tactile messages.
    """
    prev_frame = np.zeros((16, 32), dtype=np.float32)
    data_tac   = []
    num        = 0
    t1         = time.time()
    backup     = None
    flag       = False
    current    = None

    # For frequency printing every 2 messages
    freq_count = 0
    freq_tstart = time.time()

    # ---- Phase 1: gather frames for median baseline ----
    while is_running.is_set():
        if num >= median_frames:
            break
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except:
                line = ""  # skip bad read

            if len(line) < 10:
                # block boundary
                if current is not None and len(current) == 16:
                    if all(len(row) == 32 for row in current):
                        try:
                            backup = np.array(current, dtype=np.float32)
                            print("Left fps", 1/(time.time() - t1))
                            t1 = time.time()
                            data_tac.append(backup)
                            num += 1
                        except ValueError:
                            backup = None
                            print("Skipping invalid left tactile block (ValueError).")
                    else:
                        print("Skipping invalid left tactile block (rows not length 32).")

                current = []
                continue

            if current is None:
                current = []
            str_values = line.split()
            try:
                int_values = [int(val) for val in str_values]
                current.append(int_values)
            except ValueError:
                continue

    # Compute median for baseline
    if len(data_tac) == 0:
        median = np.zeros((16, 32), dtype=np.float32)
    else:
        data_tac = np.array(data_tac, dtype=np.float32)
        median   = np.median(data_tac, axis=0)
    flag = True
    print("Finish Left Initialization (median)")

    # ---- Phase 2: main loop ----
    while is_running.is_set():
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except:
                line = ""

            if len(line) < 10:
                # block boundary
                if current is not None and len(current) == 16:
                    if all(len(row) == 32 for row in current):
                        try:
                            backup = np.array(current, dtype=np.float32)
                        except ValueError:
                            backup = None
                            print("Skipping invalid left tactile block (ValueError).")
                    else:
                        backup = None
                        print("Skipping invalid left tactile block (rows not length 32).")
                current = []

                if backup is not None:
                    contact_data = backup - median - THRESHOLD
                    contact_data = np.clip(contact_data, 0, 100)
                    if np.max(contact_data) < THRESHOLD:
                        contact_data_norm = contact_data / NOISE_SCALE
                    else:
                        contact_data_norm = contact_data / np.max(contact_data)

                    if flag:
                        temp_filtered_data = temporal_filter(contact_data_norm, prev_frame, alpha=0.5)
                        prev_frame = temp_filtered_data

                        temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)
                        visualize_data = temp_filtered_data_scaled[4:]  # shape => (12,32)
                        final_12x32 = visualize_data.astype(np.float32) / 255.0

                        #print("left",final_12x32)

                        ring_buffer.put({
                            "timestamp": time.time() + receive_latency,
                            "frame": final_12x32
                        })

                        '''
                        # ----------------------
                        # FREQUENCY PRINT LOGIC
                        # ----------------------
                        freq_count += 1
                        if freq_count == 2:
                            now = time.time()
                            dt = now - freq_tstart  # time for 2 frames
                            freq = 2.0 / dt
                            print(f"[Left Tactile] Frequency ~ {freq:.2f} Hz")
                            freq_count = 0
                            freq_tstart = now
                        '''
                backup = None
                continue

            # else line is part of the block
            if current is None:
                current = []
            str_values = line.split()
            try:
                int_values = [int(val) for val in str_values]
                current.append(int_values)
            except ValueError:
                continue


class TactileControllerLeft:
    """
    Separate controller for the left tactile sensor only.
    """

    def __init__(self,
                 shm_manager,
                 port_left="/dev/ttyUSB0",
                 baud=2000000,
                 median_samples=30,
                 ring_buffer_size=200,
                 receive_latency=0.05):
        self.ser_left = serial.Serial(port_left, baud)
        time.sleep(0.2)

        example_entry = {
            "timestamp": 0.0,
            "frame": np.zeros((12,32), dtype=np.float32)
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_entry,
            get_max_k=ring_buffer_size,
            get_time_budget=0.2,
            put_desired_frequency=150
        )

        self.median_samples = median_samples
        self.receive_latency = receive_latency
        self.is_running = threading.Event()
        self.is_running.set()

        self.thread_left = threading.Thread(
            target=readThreadLeft,
            args=(self.ser_left, self.ring_buffer, self.median_samples, self.is_running, self.receive_latency),
            daemon=True
        )

    def start(self):
        self.thread_left.start()

    def stop(self):
        self.is_running.clear()
        time.sleep(0.1)
        try:
            self.ser_left.close()
        except:
            pass

    def get_all(self):
        return self.ring_buffer.get_all()

    def get(self, k=1, out=None):
        return self.ring_buffer.get_last_k(k, out=out)
