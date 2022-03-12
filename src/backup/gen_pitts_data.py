"""This script generates data for the Pittsburgh dataset
It can merge different parts of different trajectories into one folder.

Author: Haowen Lai
"""

import argparse
import pathlib
from typing import List
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Pittsburgh Dataset")
    parser.add_argument(
        "--source-dir",
        help="path to the original pittsburgh dataset trajectory folder",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="path to hold the converted dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--submap-start",
        help="Which submap id to start (with)",
        metavar="submap_st",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--submap-end",
        help="Which submap id to end (with)",
        metavar="submap_end",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--interval",
        default=4,
        help="distance in meters between two samples",
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="seed for random operation",
        type=int,
    )

    args = parser.parse_args()
    return args


def gen_train_data(
    src_dirs: List[str],
    save_dir: str,
    submap_st_ids: List[int],
    submap_end_ids: List[int],
    interval: int,
):
    """Generate data for training. No need for random operation."""
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_submap_id = 1

    # may combine multiple trajectories into one output trajectory
    for src_dir, submap_st_id, submap_end_id in zip(
        src_dirs, submap_st_ids, submap_end_ids
    ):
        # for each source trajectory
        poses_6d = []
        for src_submap_id in range(submap_st_id, submap_end_id, interval):
            # read, down sample and save
            pcd = o3d.io.read_point_cloud(f'{src_dir}/{src_submap_id:06d}.pcd')
            pcd = pcd.voxel_down_sample(voxel_size=0.35)
            o3d.io.write_point_cloud(f'{save_dir}/{save_submap_id:06d}.pcd', pcd)

            # load pose 6d
            poses_6d.append(np.load(f'{src_dir}/{src_submap_id:06d}_pose6d.npy')[:6])
            save_submap_id += 1

        # save pose 6d
        poses_6d = np.vstack(poses_6d)
        np.save(f'{save_dir}/poses_6d.npy', poses_6d)


def gen_test_data(
    src_dirs: List[str],
    save_dir: str,
    submap_st_ids: List[int],
    submap_end_ids: List[int],
    interval: int,
    seed: int,
):
    """Generate data for testing. We use the random operation to mix
    the reference submaps and test submaps together.
        The submaps determined by `src_dirs`, `submap_st_ids`, and
    `submap_end_ids` are used both as reference and test. The number
    of submaps for reference and testing is the same.
        For the testing submaps, we randomly translates points along
    the x- and y- axis in range [-2.0m, 2.0m), and randomly rotates
    points around the z-axis in range [-pi/6, pi/6).
    """
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed=seed)  # set random seed

    # divide into reference and test files
    ref_files = []
    for src_dir, submap_st_id, submap_end_id in zip(
        src_dirs, submap_st_ids, submap_end_ids
    ):
        files = [
            f'{src_dir}/{submap_id:06d}.pcd'
            for submap_id in range(submap_st_id, submap_end_id, interval)
        ]
        ref_files.extend(files)

    num_ref_files = len(ref_files)
    random_ids = np.random.permutation(2 * num_ref_files)

    # start to process...
    save_submap_id = 1
    for random_id in random_ids:
        if random_id < num_ref_files:  # ref, no ramdom translation and rotations
            # read, down sample and save
            pcd = o3d.io.read_point_cloud(ref_files[random_id])
            pcd = pcd.voxel_down_sample(voxel_size=0.35)
            o3d.io.write_point_cloud(f'{save_dir}/{save_submap_id:06d}.pcd', pcd)
        else:  # ref, need ramdom translation and rotations
            random_id %= num_ref_files

            # random transformation
            trans = np.identity(4)
            trans[:3, :3] = R.from_euler(
                'xyz',
                (0, 0, (np.random.rand() - 0.5) * np.pi / 3),
            ).as_matrix()
            trans[:3, 3] = [
                (np.random.rand() - 0.5) * 4,
                (np.random.rand() - 0.5) * 4,
                0,
            ]

            # read, down sample, transform and save
            pcd = o3d.io.read_point_cloud(ref_files[random_id])
            pcd = pcd.voxel_down_sample(voxel_size=0.35)
            pcd.transform(trans)
            o3d.io.write_point_cloud(f'{save_dir}/{save_submap_id:06d}.pcd', pcd)

        save_submap_id += 1


def main():
    args = parse_arguments()

    if args.seed is None:
        print('processing training data...')
        gen_train_data(
            args.source_dir,
            args.save_dir,
            args.submap_start,
            args.submap_end,
            args.interval,
        )
    else:
        print('processing testing data...')
        gen_test_data(
            args.source_dir,
            args.save_dir,
            args.submap_start,
            args.submap_end,
            args.interval,
            args.seed,
        )
    print('Done.')


if __name__ == '__main__':
    main()
