"""This script convert the original Pittsburgh Dataset to the kapture format.
For help in usage, please run
    ~$ python convert_to_kapture.py -h
"""

import argparse
from pathlib import Path
import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Kapture Format Dataset")
    parser.add_argument(
        "--source-dir",
        help="path to the original pittsburgh dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="path to hold the converted dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--slice-size",
        default=50.0,
        metavar="r",
        help="local map generation size (m)",
        type=float,
    )
    parser.add_argument(
        "--traj-start",
        default=1,
        metavar="traj_st",
        help="Which trajectory to start (with)",
        type=int,
    )
    parser.add_argument(
        "--traj-end",
        default=55,
        metavar="traj_end",
        help="Which trajectory to end (with)",
        type=int,
    )
    parser.add_argument(
        "--interval",
        default=2.0,
        help="distance in meters between two samples",
        type=float,
    )
    args = parser.parse_args()
    return args


def process_one_trajectory(
    ori_traj_data_path: Path,
    new_traj_base_path: Path,
    slice_size: float,
    interval: float,
):
    """Process one trajectory and add it to kapture format.
    Args:
        ori_traj_data_path: original folder for this trajectory
        new_traj_base_path: new folder to hold this trajectory
        slice_size: the submap has size slice_size*slice_size
        interval: the distance between every two submaps
    """
    sensors_path = new_traj_base_path / Path('sensors')
    records_data_path = sensors_path / Path('records_data')
    record_lidar_file = sensors_path / Path('records_lidar.txt')
    trajectories_file = sensors_path / Path('trajectories.txt')
    kapture_format_str = '# kapture format: 1.1\n'

    # write general info
    sensors_path.mkdir(parents=True, exist_ok=True)
    with open(sensors_path / Path('sensors.txt'), 'w') as f_sensors:
        f_sensors.write(kapture_format_str)
        f_sensors.write('lidar0, lidar, lidar\n')

    with open(sensors_path / Path('rigs.txt'), 'w') as f_rigs:
        f_rigs.write(kapture_format_str)
        f_rigs.write('rig, lidar0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n')

    f_traj = open(trajectories_file, 'w')  # file object, remember to close
    f_lidar = open(record_lidar_file, 'w')  # file object, remember to close
    f_traj.write(kapture_format_str)
    f_lidar.write(kapture_format_str)

    # load original data
    records_data_path.mkdir(exist_ok=True)
    poses_6d = np.loadtxt(
        ori_traj_data_path / Path('poses.txt'),
        delimiter=' ',
        usecols=(0, 1, 2, 3, 4, 5),
    )  # x, y, z, roll, pitch, yaw
    map_pcd = o3d.io.read_point_cloud(str(ori_traj_data_path / Path('cloudGlobal.pcd')))

    frame_id = 0
    last_xyz = np.array([9.9e9, 9.9e9, 9.9e9])  # a large number
    bbox_crop = o3d.geometry.AxisAlignedBoundingBox(
        np.array([-slice_size, -slice_size, -100.0]),
        np.array([slice_size, slice_size, 100.0]),
    )  # for cropping submaps

    for pose_6d in poses_6d:
        # judge interval distance
        if np.linalg.norm(pose_6d[:3] - last_xyz, ord=2) < interval:
            continue

        # the pose for current submap
        rot = o3d.geometry.get_rotation_matrix_from_zyx(pose_6d[-1:2:-1])
        trans = np.identity(4, np.float64)
        trans[0:3, 0:3] = rot
        trans[:3, 3] = pose_6d[:3]

        # transform the global point cloud to the origin, crop, and save
        map_trans_pcd = copy.deepcopy(map_pcd)
        map_trans_pcd.transform(np.linalg.inv(trans))
        map_trans_pcd = map_trans_pcd.crop(bbox_crop)  # crop
        pcd_path = records_data_path / Path(f'{frame_id:06d}.pcd')
        o3d.io.write_point_cloud(str(pcd_path), map_trans_pcd)

        # add submap name and path to records_lidar.txt file
        submap_record = f'{frame_id}, lidar0, {pcd_path.name}\n'
        f_lidar.write(submap_record)

        # add trajectory pose info
        rot_quat = Rotation.from_matrix(rot).as_quat()  # qx, qy, qz, qw
        pose_record = (
            f'{frame_id}, rig, {rot_quat[3]}, {rot_quat[0]}, {rot_quat[1]}, '
            f'{rot_quat[2]}, {pose_6d[0]}, {pose_6d[1]}, {pose_6d[2]}\n'
        )
        f_traj.write(pose_record)

        last_xyz = pose_6d[:3]
        frame_id += 1

    f_traj.close()
    f_lidar.close()


def main():
    args = parse_arguments()

    for traj_id in tqdm(range(args.traj_start, args.traj_end + 1), 'proc'):
        ori_traj_data_path = Path(args.source_dir) / Path(f'Train{traj_id}')
        new_traj_base_path = Path(args.save_dir) / Path(f'traj{traj_id}')

        process_one_trajectory(
            ori_traj_data_path,
            new_traj_base_path,
            args.slice_size,
            args.interval,
        )


if __name__ == "__main__":
    main()
