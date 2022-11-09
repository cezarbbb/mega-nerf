import glob
import os
from argparse import Namespace
from pathlib import Path

import cv2
import torch
from render_images import _render_images
from mega_nerf.opts import get_opts_base


def _get_render_videos_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--render_frames', type=int, required=True)
    parser.add_argument('--video_name', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--centroids_path', type=str, required=True)
    parser.add_argument('--save_depth_npz', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')

    return parser.parse_args()


def _write_files(hparams: Namespace) -> None:
    dataset_path = Path(hparams.dataset_path)
    for i in range(hparams.render_frames):
        metadata_path = dataset_path / 'train' / 'metadata' / '{0:06d}.pt'.format(i)
        if metadata_path.exists():
            metadata = torch.load(metadata_path)
            input = Path(hparams.input)
            with(input / 'poses.txt').open('a') as f:
                print('writing poses.txt, the source is {}\n'.format(metadata['c2w']))
                for j in metadata['c2w']:
                    for k in j:
                        f.write('{} '.format(k))
                f.write('\n')
            with(input / 'intrinsics.txt').open('a') as f:
                print('writing intrinsics.txt, the source is {}\n'.format(metadata['intrinsics']))
                for j in metadata['intrinsics']:
                    f.write('{} '.format(j))
                f.write('\n')
            with(input / 'embeddings.txt').open('a') as f:
                print('writing embeddings.txt, the source is {}\n'.format(i))
                f.write('{}\n'.format(i))
    print('all files have been written.\n')


def _generate_video(hparams: Namespace) -> None:
    print('start generating video.\n')
    output = Path(hparams.output)
    images_path = output / 'rgbs'
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(hparams.video_name, fourcc, fps, (745, 745))
    images = glob.glob(os.path.join(images_path, '*.jpg'))

    for i in range(len(images)):
        image_name = os.path.join(images_path, '{0:06d}.jpg'.format(i))
        print('image {} loaded.\n'.format(image_name))
        frame = cv2.imread(image_name)
        video_writer.write(frame)

    video_writer.release()
    print('video has been successfully generated.\n')


def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None

    _write_files(hparams)

    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            _render_images(hparams)
    else:
        _render_images(hparams)

    _generate_video(hparams)


if __name__ == '__main__':
    main(_get_render_videos_opts())

