#!/usr/bin/env python3
"""
Combined SLAM pipeline:
  (0) 00_process_videos.py
  (1) 01_extract_gopro_imu.py
  (2) 02_cut_and_match.py
  (3) 02_create_map.py
  (4) 03_batch_slam.py
  (5) 04_detect_aruco.py
  (6) 05_run_calibrations.py
  (7) 06_generate_dataset_plan.py

Usage:
    python run_slam_pipeline.py [OPTIONS] <session_dir>...
"""

import sys
import os
import pathlib
import subprocess
import click

@click.command()
@click.argument('session_dir', nargs=-1, type=click.Path(exists=True))
@click.option('--bag', required=True, help="Path to tactile JSON file.")
@click.option('-c', '--calibration_dir', type=click.Path(exists=True), default=None,
              help="(Optional) path to directory with camera intrinsics/aruco_config.")
@click.option('--plot_video', is_flag=True, default=False,
              help="If set, create overlay videos in cut & match step.")
@click.option('--plot_images', is_flag=True, default=False,
              help="If set, create overlay images in cut & match step.")
@click.option('--cut_delay', type=float, default=1.3,
              help="Seconds after black frame to cut.")
@click.option('--qr_latency', type=float, default=0.09,
              help="Latency offset for QR code.")
@click.option('--black_threshold', type=float, default=10.0,
              help="Threshold for black frame detection.")
def main(session_dir, bag, calibration_dir, plot_video, plot_images,
         cut_delay, qr_latency, black_threshold):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')

    if calibration_dir:
        calib_dir = pathlib.Path(calibration_dir)
    else:
        calib_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    if not calib_dir.is_dir():
        click.echo(f"Warning: calibration dir not found at {calib_dir}")
    camera_intrinsics = calib_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calib_dir.joinpath('aruco_config.yaml')

    for session in session_dir:
        sdir = pathlib.Path(os.path.expanduser(session)).absolute()
        demos_dir = sdir.joinpath('demos')

        # Step 0: Process videos
        click.echo("############## Step 0: 00_process_videos #############")
        script_00 = script_dir.joinpath('00_process_videos.py')
        assert script_00.is_file(), "Missing 00_process_videos.py"
        cmd = [sys.executable, str(script_00), str(sdir)]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 1: Extract GoPro IMU
        click.echo("############## Step 1: 01_extract_gopro_imu #############")
        script_01 = script_dir.joinpath('01_extract_gopro_imu.py')
        assert script_01.is_file(), "Missing 01_extract_gopro_imu.py"
        cmd = [sys.executable, str(script_01), str(sdir)]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 2: Cut & tactile match
        click.echo("############## Step 2: 02_cut_and_match #############")
        cut_script = script_dir.joinpath('02_cut_and_match.py')
        assert cut_script.is_file(), f"{cut_script} not found!"
        # Collect videos to process
        videos_to_process = []
        if demos_dir.is_dir():
            for sub in sorted(demos_dir.iterdir()):
                if not sub.is_dir() or not sub.name.startswith('demo'):
                    continue
                vid = sub.joinpath('raw_video.mp4')
                tactile_npy = sub.joinpath('tactile.npy')
                if vid.exists() and not tactile_npy.exists():
                    videos_to_process.append(str(vid))
        if not videos_to_process:
            click.echo("No raw_video.mp4 files to process. Skipping cut & match.")
        else:
            cmd = [
                sys.executable, str(cut_script),
                '--bag', str(bag),
                '--cut_delay', str(cut_delay),
                '--qr_latency', str(qr_latency),
                '--black_threshold', str(black_threshold)
            ]
            if plot_video:
                cmd.append('--plot_video')
            if plot_images:
                cmd.append('--plot_images')
            cmd.append('--video')
            cmd.extend(videos_to_process)
            click.echo(f"Running single cut & match for {len(videos_to_process)} videos")
            subprocess.run(cmd, check=True)

        # Step 3: Create map
        click.echo("########## Step 3: 02_create_map ##########")
        mapping_dir = demos_dir.joinpath('mapping')
        map_path = mapping_dir.joinpath('map_atlas.osa')
        create_map_script = script_dir.joinpath('02_create_map.py')
        assert create_map_script.is_file(), "Missing 02_create_map.py"
        if not map_path.is_file():
            cmd = [
                sys.executable, str(create_map_script),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path)
            ]
            click.echo(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            click.echo(f"Map already exists at {map_path}, skipping creation.")

        # Step 4: Batch SLAM
        click.echo("########## Step 4: 03_batch_slam ##########")
        batch_slam_script = script_dir.joinpath('03_batch_slam.py')
        assert batch_slam_script.is_file(), "Missing 03_batch_slam.py"
        cmd = [
            sys.executable, str(batch_slam_script),
            '--input_dir', str(demos_dir),
            '--map_path', str(map_path)
        ]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 5: Detect ArUco
        click.echo("########## Step 5: 04_detect_aruco ##########")
        detect_script = script_dir.joinpath('04_detect_aruco.py')
        assert detect_script.is_file(), "Missing 04_detect_aruco.py"
        cmd = [
            sys.executable, str(detect_script),
            '--input_dir', str(demos_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 6: Run calibrations
        click.echo("########## Step 6: 05_run_calibrations ##########")
        calib_script = script_dir.joinpath('05_run_calibrations.py')
        assert calib_script.is_file(), "Missing 05_run_calibrations.py"
        cmd = [sys.executable, str(calib_script), str(sdir)]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 7: Generate dataset plan
        click.echo("########## Step 7: 06_generate_dataset_plan ##########")
        plan_script = script_dir.joinpath('06_generate_dataset_plan.py')
        assert plan_script.is_file(), "Missing 06_generate_dataset_plan.py"
        cmd = [
            sys.executable, str(plan_script),
            '--input', str(sdir)
        ]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    click.echo("All sessions completed.")


if __name__ == "__main__":
    main()
