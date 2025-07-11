#!/usr/bin/env python3
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

# %%
@click.command(help='Session directories. Assuming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hard-coded subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')

        # create raw_videos if missing and move all MP4s inside
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir didn't exist — created and moved MP4s in.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                if 'overlay' in mp4_path.name.lower():
                    print(f"Skipping overlay video: {mp4_path.name}")
                    continue
                shutil.move(mp4_path, input_dir / mp4_path.name)

        # find all MP4s now residing under raw_videos
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if 'overlay' in mp4_path.name.lower():
                    print(f"Skipping overlay video: {mp4_path.name}")
                    continue
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # create destination directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)

                # move video & rename to raw_video.mp4
                out_video_path = this_out_dir / 'raw_video.mp4'
                shutil.move(mp4_path, out_video_path)

                # create symlink back at original location
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                mp4_path.symlink_to(os.path.join(dots, rel_path))

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
