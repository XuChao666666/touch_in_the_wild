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
        session = pathlib.Path(os.path.expanduser(session)).absolute()  # 转为绝对路径
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

        # find all MP4s now residing under raw_videos 找到位于raw_videos文件夹下的MP4文件
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                # 过滤两种不需要处理的文件：1、文件中有'overlay'（忽略大小写）；2、’判断文件是否为符号链接（软链接）
                if 'overlay' in mp4_path.name.lower():
                    print(f"Skipping overlay video: {mp4_path.name}")
                    continue
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))  # 获取相机序列号
                meta = list(et.get_metadata(str(mp4_path)))[0]      # 获取视频数据
                cam_serial = meta['QuickTime:CameraSerialNumber']   # 获取相机序列号
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")   # 生成一个格式化的输出目录名称

                # create destination directory 构造输出目录名
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)

                # move video & rename to raw_video.mp4 移动并重命名视频文件 
                out_video_path = this_out_dir / 'raw_video.mp4'
                shutil.move(mp4_path, out_video_path)

                # create symlink back at original location 创建回链 
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                mp4_path.symlink_to(os.path.join(dots, rel_path))

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
