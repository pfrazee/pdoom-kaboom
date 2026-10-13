#!/usr/bin/env python3
"""Render the latest performance from the bundled song and timing data."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end', type=float, default=151.6, help='End time in seconds')
    parser.add_argument('--still', type=float, help='Render one JPEG at this song time')
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 151.6:
        parser.error('Use 0 <= start < end <= 151.6')
    if args.still is not None and not 0 <= args.still < 151.6:
        parser.error('Still time must be between 0 and 151.6 seconds')
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg must be installed and available on PATH')

    sys.path.insert(0, str(ROOT / 'src'))
    import create_video_v3 as video

    if args.still is not None:
        destination = video.OUT / f'frame-{args.still:g}.jpg'
        video.render(args.still).save(destination, quality=95)
        print(destination)
        return

    # Snap both bounds to the renderer's frame grid so the audio stays synchronized.
    start = round(args.start * video.FPS) / video.FPS
    end = round(args.end * video.FPS) / video.FPS
    if end <= start:
        parser.error('The selected interval must contain at least one frame')
    complete = start == 0 and end == video.DURATION
    name = 'P-doom-Kaboom-v3' if complete else f'preview-{start:g}-{end:g}'
    silent = video.WORK / f'{name}-silent.mp4'
    destination = video.OUT / f'{name}.mp4'
    video.render_segment(start, end, silent)
    command = [
        'ffmpeg', '-y', '-v', 'error', '-i', str(silent),
        '-ss', str(start), '-i', str(ROOT / 'assets/audio/P(doom)! Kaboom!.m4a'),
    ]
    if complete:
        command += ['-i', str(ROOT / 'data/v3-chapters.ffmeta')]
    command += ['-map', '0:v:0', '-map', '1:a:0']
    command += ['-map_metadata', '2', '-map_chapters', '2'] if complete else ['-map_metadata', '-1', '-map_chapters', '-1']
    command += [
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        '-t', str(end - start), '-movflags', '+faststart', str(destination),
    ]
    subprocess.run(command, check=True)
    if complete:
        subtitles = '\n\n'.join(
            f"{i + 1}\n{video.base.ts(c['start'])} --> {video.base.ts(c['end'])}\n{c['text']}"
            for i, c in enumerate(video.CUES)
        ) + '\n'
        (video.OUT / f'{name}.srt').write_text(subtitles, encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
