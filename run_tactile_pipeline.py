#!/usr/bin/env python3
"""
Tactile-only pipeline
  (0) 00_process_videos.py
  (1) 02_cut_and_match.py
  (2) 03_generate_dataset_plan_tactile.py
  (3) 04_generate_replay_buffer_tactile.py

Usage:
    python run_tactile_pipeline.py [OPTIONS] <session_dir>...
"""

import sys
import os
import pathlib
import subprocess
import click

@click.command()
@click.argument('session_dir', nargs=-1, type=click.Path(exists=True))
@click.option('--bag', required=True, help="Path to tactile JSON file (for cut & match).")
@click.option('--plot_video', is_flag=True, default=False,
              help="If set, create overlay videos in cut & match.")
@click.option('--plot_images', is_flag=True, default=False,
              help="If set, create overlay images in cut & match.")
@click.option('--cut_delay', type=float, default=1.3,
              help="Seconds after black frame to cut.")
@click.option('--qr_latency', type=float, default=0.09,
              help="Latency offset for QR code.")
@click.option('--black_threshold', type=float, default=10.0,
              help="Threshold for black-frame detection.")
def main(session_dir, bag, plot_video, plot_images,
         cut_delay, qr_latency, black_threshold):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_tactile_pipeline')

    for session in session_dir:
        sdir = pathlib.Path(os.path.expanduser(session)).absolute()
        demos_dir = sdir.joinpath('demos')

        click.echo("############## Step 0: 00_process_videos #############")
        script_00 = script_dir / '00_process_videos.py'
        subprocess.run([sys.executable, str(script_00), str(sdir)], check=True)

        click.echo("############## Step 1: 01_extract_gopro_imu #############")
        script_01 = script_dir.joinpath('01_extract_gopro_imu.py')
        assert script_01.is_file(), "Missing 01_extract_gopro_imu.py"
        cmd = [sys.executable, str(script_01), str(sdir)]
        click.echo(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        click.echo("############## Step 2: 02_cut_and_match #############")
        cut_script = script_dir / '02_cut_and_match.py'

        vids_to_process = []
        if demos_dir.is_dir():
            for sub in sorted(demos_dir.iterdir()):
                if not sub.is_dir() or not sub.name.startswith('demo'):
                    continue
                vid = sub / 'raw_video.mp4'
                tactile_npy = sub / 'tactile.npy'
                if vid.exists() and not tactile_npy.exists():
                    vids_to_process.append(str(vid))

        if not vids_to_process:
            click.echo("No raw_video.mp4 files need cutting. Skipping.")
        else:
            cmd = [
                sys.executable, str(cut_script),
                '--bag', str(bag),
                '--cut_delay', str(cut_delay),
                '--qr_latency', str(qr_latency),
                '--black_threshold', str(black_threshold),
                '--video', *vids_to_process
            ]
            if plot_video:
                cmd.append('--plot_video')
            if plot_images:
                cmd.append('--plot_images')
            click.echo(f"Running cut & match on {len(vids_to_process)} videos")
            subprocess.run(cmd, check=True)

        click.echo("########## Step 3: 03_generate_dataset_plan_tactile ##########")
        plan_script = script_dir / '03_generate_dataset_plan_tactile.py'
        subprocess.run([sys.executable, str(plan_script), '--input', str(sdir)], check=True)


    click.echo("All tactile sessions completed.")


if __name__ == "__main__":
    main()
