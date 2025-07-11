#!/usr/bin/env python3
"""
generate_dataset_zarr_tactile.py

Vision + tactile only: no metadata, tag-detection, or trajectory files required.
All optional data are replaced with obvious PLACEHOLDER defaults.  A thread lock
prevents writes while flushing to disk. 
"""
import zipfile
import os
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import av.logging
import multiprocessing
import concurrent.futures
import threading
from tqdm import tqdm
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform,
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()

# ---------------------------------------------------------------------- #
#                                CLI                                     #
# ---------------------------------------------------------------------- #
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(
    input,
    output,
    out_res,
    out_fov,
    compression_level,
    no_mirror,
    mirror_swap,
    num_workers,
):

    if os.path.isfile(output):
        click.confirm(f'Output file {output} exists! Overwrite?', abort=True)

    out_res = tuple(map(int, out_res.split(',')))
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    fisheye_converter = None
    if out_fov is not None:
        intr_path = (
            pathlib.Path(os.path.expanduser(input[0]))
            .absolute()
            .joinpath('calibration', 'gopro_intrinsics_2_7k.json')
        )
        intr = json.load(open(intr_path, 'r'))
        intr = parse_fisheye_intrinsics(intr)
        fisheye_converter = FisheyeRectConverter(
            **intr,
            out_size=out_res,
            out_fov=out_fov,
        )

    out_rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = []
    store_lock = threading.RLock()
    folder_counts = {}

    for folder in input:
        folder = pathlib.Path(folder).expanduser().absolute()
        demos_path = folder / 'demos'
        plan_path = folder / 'tactile_dataset_plan.pkl'
        if not plan_path.is_file():
            print(f"[SKIP] {folder.name}: no tactile_dataset_plan.pkl")
            continue
        try:
            plan = pickle.load(open(plan_path, 'rb'))
        except Exception as e:
            print(f"[ERROR] loading {plan_path}: {e}")
            continue

        videos_dict = defaultdict(list)

        for ep in plan:
            grippers = ep['grippers']
            cameras = ep['cameras']
            n_grippers = n_grippers or len(grippers)
            n_cameras = n_cameras or len(cameras)
            ep_data = {}
            ep_len = None

            # ----- loop over cameras (tactile only) ----- #
            for cam_id, cam in enumerate(cameras):
                rel = cam['video_path']
                abs_path = demos_path / rel
                start, end = cam['video_start_end']
                tactile = abs_path.parent / 'tactile.npy'
                if not tactile.is_file():
                    print(f"[SKIP EP] {abs_path.parent.name}: tactile.npy missing")
                    ep_data = None
                    break
                try:
                    arr = np.load(tactile, allow_pickle=True)
                except Exception as e:
                    print(f"[SKIP EP] unreadable tactile.npy ({e})")
                    ep_data = None
                    break
                slice_ = arr[start:end].astype(np.float32)
                if slice_.shape[0] != (end - start):
                    print(
                        f"[SKIP EP] {abs_path.parent.name}: frame mismatch "
                        f"{end - start} vs {slice_.shape[0]}"
                    )
                    ep_data = None
                    break
                ep_len = ep_len or (end - start)
                if (end - start) != ep_len:
                    print(f"[SKIP EP] frame-count mismatch within episode")
                    ep_data = None
                    break
                ep_data[f'camera{cam_id}_tactile'] = slice_

            if ep_data is None:
                continue

            out_rb.add_episode(data=ep_data, compressors=None)

            # plan video extraction
            n_frames = None
            for cam_id, cam in enumerate(cameras):
                mp4 = demos_path / cam['video_path']
                if not mp4.is_file():
                    print(f"[SKIP VIDEO] {mp4} missing")
                    continue
                start, end = cam['video_start_end']
                n_frames = n_frames or (end - start)
                videos_dict[str(mp4)].append(
                    dict(
                        camera_idx=cam_id,
                        frame_start=start,
                        frame_end=end,
                        buffer_start=buffer_start,
                    )
                )
            buffer_start += n_frames or 0
            all_videos.update(videos_dict.keys())

        vid_args.extend(videos_dict.items())
        folder_counts[str(folder)] = len(videos_dict)
        print(f"{folder.name}: {len(videos_dict)} videos used")

    if not vid_args:
        print("No videos to process, exiting.")
        return

    '''
    total_vids = sum(folder_counts.values())
    with open('video_counts.txt', 'w') as f:
        for k, v in folder_counts.items():
            f.write(f"{k}: {v} videos\n")
        f.write(f"Total: {total_vids} videos\n")
    print("Video counts saved to video_counts.txt")
    '''

    with av.open(list(all_videos)[0]) as tmp:
        h_src, w_src = tmp.streams.video[0].height, tmp.streams.video[0].width

    img_comp = JpegXl(level=compression_level, numthreads=1)
    total_frames = buffer_start
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        out_rb.data.require_dataset(
            name=name,
            shape=(total_frames,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_comp,
            dtype=np.uint8,
        )

    def video_to_zarr(rb, mp4_path, tasks):
        with av.open(mp4_path) as cont:
            h_src, w_src = cont.streams.video[0].height, cont.streams.video[0].width
            n_frames = cont.streams.video[0].frames

        tag_detection_results = [{'tag_dict': {}} for _ in range(n_frames)]  # PLACEHOLDER
        resize_tf = get_image_transform(in_res=(w_src, h_src), out_res=out_res)
        tasks = sorted(tasks, key=lambda t: t['frame_start'])
        cam_idx = tasks[0]['camera_idx']
        arr = rb.data[f'camera{cam_idx}_rgb']

        # mirror mask (optional)
        is_mirror = None
        if mirror_swap:
            mask = np.ones((out_res[1], out_res[0], 3), np.uint8)
            mask = draw_predefined_mask(mask, color=(0, 0, 0), mirror=False, gripper=False, finger=True)
            is_mirror = mask[..., 0] == 0

        av.logging.set_level(av.logging.ERROR)

        def open_relaxed(path):
            try:
                return av.open(
                    path,
                    options=dict(
                        fflags='+discardcorrupt',
                        err_detect='ignore_err',
                        loglevel='error',
                    ),
                )
            except av.AVError as e:
                print(f"[WARN] {path} unrecoverable ({e})")
                return None

        container = open_relaxed(mp4_path)
        if container is None:
            return False

        with container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            frame_idx = 0
            task_ptr = 0
            for packet in tqdm(
                container.demux(stream), total=stream.frames, leave=False, disable=True
            ):
                try:
                    frames = packet.decode()
                except av.AVError:
                    frame_idx += 1
                    continue

                for frame in frames:
                    if task_ptr >= len(tasks):
                        break
                    task = tasks[task_ptr]
                    if frame_idx < task['frame_start']:
                        frame_idx += 1
                        continue
                    if frame_idx >= task['frame_end']:
                        task_ptr += 1
                        frame_idx += 1
                        continue

                    img = frame.to_ndarray(format='rgb24')
                    # inpaint placeholder tags — list is empty
                    img = draw_predefined_mask(
                        img,
                        color=(0, 0, 0),
                        mirror=no_mirror,
                        gripper=False,
                        finger=False,
                    )
                    img = fisheye_converter.forward(img) if fisheye_converter else resize_tf(img)
                    if mirror_swap and is_mirror is not None:
                        img[is_mirror] = img[:, ::-1, :][is_mirror]

                    with store_lock:
                        arr_idx = task['buffer_start'] + (frame_idx - task['frame_start'])
                        arr[arr_idx] = img
                    frame_idx += 1
        return True

    total_videos = len(vid_args)
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut = {ex.submit(video_to_zarr, out_rb, mp4, t): mp4 for mp4, t in vid_args}
        for f in tqdm(concurrent.futures.as_completed(fut), total=total_videos, desc="Videos"):
            p = fut[f]
            try:
                ok = f.result()
            except Exception as e:
                print(f"[ERROR] {p}: {e}")
                ok = False
            if ok:
                processed += 1
                if processed % 100 == 0:
                    with store_lock:
                        with zarr.ZipStore(output, mode='w', compression=zipfile.ZIP_STORED) as st:
                            out_rb.save_to_store(st)
                    print(f"→ autosaved after {processed} / {total_videos} videos")

    # final save
    with store_lock:
        with zarr.ZipStore(output, mode='w', compression=zipfile.ZIP_STORED) as st:
            out_rb.save_to_store(st)
    print(f"Done — {processed}/{total_videos} videos saved to {output}")


if __name__ == '__main__':
    main()
